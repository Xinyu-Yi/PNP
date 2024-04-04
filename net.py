import torch
import articulate as art
from articulate.utils.torch.rnn import *
from dynamics import PhysicsOptimizer


class NumericalDifferentiation:
    def __init__(self, dt=1., n_points=2):
        assert n_points in [2, 3, 5]
        self.dt = dt
        self.n_points = n_points
        self.reset()

    def reset(self):
        self.x = [None] * (self.n_points - 1)

    def __call__(self, x):
        if self.x[0] is None:
            d = torch.zeros_like(x)
        elif self.n_points == 2:
            d = (x - self.x[0]) / self.dt
        elif self.n_points == 3:
            d = (3 * x - 4 * self.x[1] + self.x[0]) / (2 * self.dt)
        elif self.n_points == 5:
            d = (25 * x - 48 * self.x[3] + 36 * self.x[2] - 16 * self.x[1] + 3 * self.x[0]) / (12 * self.dt)
        self.x = self.x[1:] + [x.clone()]
        return d


class PNP(torch.nn.Module):
    name = 'PNP'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    ji_reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ji_ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    def __init__(self):
        super(PNP, self).__init__()
        self.plnet_A = torch.nn.Sequential(
            torch.nn.Linear(108, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(512, 15),
        )
        self.plnet_net1 = RNNWithInit(input_linear=False,
                                      input_size=60,
                                      output_size=15,
                                      hidden_size=512,
                                      num_rnn_layer=3,
                                      dropout=0.4)
        self.iknet_net1 = RNN(input_linear=False,
                              input_size=45 + 15,
                              output_size=69,
                              hidden_size=512,
                              num_rnn_layer=3,
                              dropout=0.4)
        self.iknet_net2 = RNN(input_linear=False,
                              input_size=45 + 69,
                              output_size=90,
                              hidden_size=512,
                              num_rnn_layer=3,
                              dropout=0.4)
        self.vrnet_net1 = RNN(input_linear=False,
                              input_size=144,
                              output_size=72,
                              hidden_size=512,
                              num_rnn_layer=3,
                              dropout=0.4)
        self.vrnet_net2 = RNN(input_linear=False,
                              input_size=144,
                              output_size=2,
                              hidden_size=512,
                              num_rnn_layer=3,
                              dropout=0.4)

        self.to(self.device).load_state_dict(torch.load('data/weights/PNP/weights.pt', map_location=self.device))
        self.body_model = art.ParametricModel('models/SMPL_male.pkl', vert_mask=self.vi_mask, device=self.device)
        self.dynamics_optimizer = PhysicsOptimizer(debug=False, quiet=False)
        self.nd = NumericalDifferentiation(dt=1/60)
        self.nd2 = NumericalDifferentiation(dt=1/60)
        self.rnn_initialize()  # using T-pose
        self.eval()

    @torch.no_grad()
    def rnn_initialize(self, init_pose=None):
        if init_pose is None:
            init_pose = torch.eye(3, device=self.device).expand(1, 24, 3, 3)
        else:
            init_pose = init_pose.view(1, 24, 3, 3).to(self.device)
            init_pose[0, 0] = torch.eye(3, device=self.device)
        pl = self.body_model.forward_kinematics(init_pose, calc_mesh=True)[2].view(6, 3)
        pl = (pl[:5] - pl[5:]).ravel()
        self.pl1hc = [_.contiguous() for _ in self.plnet_net1.init_net(pl).view(1, 2, self.plnet_net1.num_layers, self.plnet_net1.hidden_size).permute(1, 2, 0, 3)]
        self.ik1hc = None
        self.ik2hc = None
        self.vr1hc = None
        self.vr2hc = None
        self.pRB_dyn = pl.view(5, 3, 1)
        self.dynamics_optimizer.reset_states()
        self.nd.reset()
        self.nd2.reset()

    @torch.no_grad()
    def forward_frame(self, a, w, R):
        RIR = R[5]
        aRB_sta = RIR.t().matmul(a.view(6, 3, 1))
        wRB_sta = RIR.t().matmul(w.view(6, 3, 1))
        RRB_sta = RIR.t().matmul(R)
        wRR_sta = wRB_sta[5:]
        wRR_sta_dot = self.nd(wRR_sta)
        vRB_dyn = self.nd2(self.pRB_dyn)

        # PL-A
        x = torch.cat((aRB_sta.ravel() / 20, RRB_sta.ravel(), wRR_sta.ravel() / 4, wRR_sta_dot.ravel() / 400, vRB_dyn.ravel() / 2, self.pRB_dyn.ravel())).view(1, -1)  # 1 subject
        x = self.plnet_A(x) * 5

        RRB_dyn = RRB_sta[:5]
        aRB_dyn = aRB_sta[:5] - aRB_sta[5:] - x.view(5, 3, 1)
        imu_dyn = torch.cat((aRB_dyn.ravel() / 20, RRB_dyn.ravel())).view(1, -1)  # 1 subject

        # PL-s1
        x = imu_dyn
        x, self.pl1hc = self.plnet_net1.rnn(x.unsqueeze(0), self.pl1hc)
        x = self.plnet_net1.linear2(x.squeeze(0))
        pRB_dyn = x.view(-1, 5, 3)
        self.pRB_dyn = pRB_dyn.view(5, 3, 1)

        # IK-s1
        x = torch.cat((RRB_dyn.view(1, -1), pRB_dyn.flatten(1)), dim=1)
        x, self.ik1hc = self.iknet_net1.rnn(x.unsqueeze(0), self.ik1hc)
        x = self.iknet_net1.linear2(x.squeeze(0))
        jpos = x.view(-1, 23, 3)

        # IK-s2
        x = torch.cat((RRB_dyn.view(1, -1), jpos.flatten(1)), dim=1)
        x, self.ik2hc = self.iknet_net2.rnn(x.unsqueeze(0), self.ik2hc)
        x = self.iknet_net2.linear2(x.squeeze(0))

        # get pose estimation
        reduced_glb_pose = art.math.r6d_to_rotation_matrix(x).view(1, 15, 3, 3)
        glb_pose = torch.eye(3, device=self.device).repeat(1, 24, 1, 1)
        glb_pose[:, self.ji_reduced] = reduced_glb_pose
        pose = self.body_model.inverse_kinematics_R(glb_pose).view(24, 3, 3)
        pose[self.ji_ignored] = torch.eye(3, device=self.device)
        pose[0] = RIR

        joint = self.body_model.forward_kinematics(pose.view(1, 24, 3, 3).to(self.device))[1].view(24, 3)
        aj = joint[1:].mm(RIR)
        imu = torch.cat((aRB_sta.ravel() / 20, RRB_sta.ravel(), wRR_sta.ravel() / 4, aj.ravel()))

        # VR-s1
        x, self.vr1hc = self.vrnet_net1.rnn(imu.unsqueeze(0), self.vr1hc)
        x = self.vrnet_net1.linear2(x.squeeze(0))
        av = x.view(24, 3) * 2

        # VR-s2
        x, self.vr2hc = self.vrnet_net2.rnn(imu.unsqueeze(0), self.vr2hc)
        x = self.vrnet_net2.linear2(x.squeeze(0))
        c = x.view(2)

        # physics-based optimization
        av = av.mm(RIR.t())
        pose_opt, tran_opt = self.dynamics_optimizer.optimize_frame(pose.cpu(), av.cpu(), c.cpu(), a.cpu(), return_grf=False)
        return pose_opt, tran_opt
