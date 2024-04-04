import torch
import tqdm
import articulate as art
import matplotlib.pyplot as plt
from net import PNP


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReducedPoseEvaluator:
    names = ['SIP Error (deg)', 'Angle Error (deg)', 'Joint Error (cm)', 'Vertex Error (cm)', 'Jitter Error (km/s^3)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator('models/SMPL_male.pkl', joint_mask=torch.tensor([1, 2, 16, 17]), device=device)
        self.ignored_joint_mask = torch.tensor([0, 7, 8, 10, 11, 20, 21, 22, 23])

    def __call__(self, pose_p, pose_t, tran_p, tran_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, self.ignored_joint_mask] = torch.eye(3, device=pose_p.device)
        pose_t[:, self.ignored_joint_mask] = torch.eye(3, device=pose_t.device)
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t)
        return torch.stack([errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 1000])


class FullPoseEvaluator:
    names = ['Absolute Jitter Error (km/s^3)']

    def __init__(self):
        self._base_motion_loss_fn = art.FullMotionEvaluator('models/SMPL_male.pkl', device=device)

    def __call__(self, pose_p, pose_t, tran_p, tran_t):
        errs = self._base_motion_loss_fn(pose_p=pose_p, pose_t=pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[4] / 1000])


def compare_realimu(data, dataset_name='', evaluate_pose=True, evaluate_tran=False):
    print('======================= Testing on %s Real Dataset =======================' % dataset_name)
    reduced_pose_evaluator = ReducedPoseEvaluator()
    full_pose_evaluator = FullPoseEvaluator()
    g = torch.tensor([0, -9.8, 0])
    nets = {
        'PNP (ours)': PNP().eval().to(device)
    }
    pose_errors = {k: [] for k in nets.keys()}
    tran_errors = {k: {window_size: [] for window_size in list(range(1, 8))} for k in nets.keys()}

    for seq_idx in range(len(data['pose'])):
        aS = data['aS'][seq_idx]
        wS = data['wS'][seq_idx]
        mS = data['mS'][seq_idx]
        RIS = data['RIS'][seq_idx]
        RIM = data['RIM'][seq_idx]
        RSB = data['RSB'][seq_idx]
        tran = data['tran'][seq_idx]
        pose = data['pose'][seq_idx]

        RMB = RIM.transpose(1, 2).matmul(RIS).matmul(RSB).to(device)
        aM = (RIM.transpose(1, 2).matmul(RIS).matmul(aS.unsqueeze(-1)).squeeze(-1) + g).to(device)
        wM = RIM.transpose(1, 2).matmul(RIS).matmul(wS.unsqueeze(-1)).squeeze(-1).to(device)
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)

        for net in nets.values():
            net.rnn_initialize(pose[0])
            net.pose_prediction = torch.zeros_like(pose)
            net.tran_prediction = torch.zeros_like(tran)

        for i in tqdm.trange(pose.shape[0]):
            for net in nets.values():
                net.pose_prediction[i], net.tran_prediction[i] = net.forward_frame(aM[i], wM[i], RMB[i])

        if evaluate_pose:
            print('[%3d/%3d  pose]' % (seq_idx, len(data['pose'])), end='')
            for k in nets.keys():
                e1 = reduced_pose_evaluator(nets[k].pose_prediction, pose, nets[k].tran_prediction, tran)
                e2 = full_pose_evaluator(nets[k].pose_prediction, pose, nets[k].tran_prediction, tran)
                pose_errors[k].append(torch.cat((e1, e2), dim=0))
                print('\t%s: %5.2fcm' % (k, pose_errors[k][-1][2, 0]), end=' ')  # joint position error
            print('')

        if evaluate_tran:
            print('[%3d/%3d  tran]' % (seq_idx, len(data['pose'])), end='')

            # compute gt move distance at every frame
            move_distance_t = torch.zeros(tran.shape[0])
            v = (tran[1:] - tran[:-1]).norm(dim=1)
            for j in range(len(v)):
                move_distance_t[j + 1] = move_distance_t[j] + v[j]

            for k in nets.keys():
                for window_size in tran_errors[k].keys():
                    # find all pairs of start/end frames where gt moves `window_size` meters
                    frame_pairs = []
                    start, end = 0, 1
                    while end < len(move_distance_t):
                        if move_distance_t[end] - move_distance_t[start] < window_size:
                            end += 1
                        else:
                            if len(frame_pairs) == 0 or frame_pairs[-1][1] != end:
                                frame_pairs.append((start, end))
                            start += 1

                    # calculate mean distance error
                    errs = []
                    for start, end in frame_pairs:
                        vel_p = nets[k].tran_prediction[end] - nets[k].tran_prediction[start]
                        vel_t = tran[end] - tran[start]
                        errs.append((vel_t - vel_p).norm() / (move_distance_t[end] - move_distance_t[start]) * window_size)
                    if len(errs) > 0:
                        tran_errors[k][window_size].append(sum(errs) / len(errs))
                print('\t%s: %5.2fm' % (k, tran_errors[k][7][-1]), end=' ')
            print('')

    print('======================= Results on %s Real Dataset =======================' % dataset_name)
    if evaluate_pose:
        print('Metrics: ', reduced_pose_evaluator.names + full_pose_evaluator.names)
        for net_name, error in pose_errors.items():
            error = torch.stack(error).mean(dim=0)
            print(net_name, end='\t')
            for error_item in error:
                print('%.2f±%.2f' % (error_item[0], error_item[1]), end='\t')  # mean & std
            print('')
    if evaluate_tran:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        plt.figure(dpi=200)
        plt.grid(linestyle='-.')
        plt.xlabel('Real travelled distance (m)', fontsize=16)
        plt.ylabel('Mean translation error (m)', fontsize=16)
        plt.title('Cumulative Translation Error', fontsize=18)
        for net_name in tran_errors.keys():
            plt.plot([0] + [_ for _ in tran_errors[net_name].keys()], [0] + [torch.tensor(_).mean() for _ in tran_errors[net_name].values()], label=net_name)
        plt.legend(fontsize=15)
        plt.show()


if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    data = torch.load('data/test_datasets/totalcapture_officalib.pt')
    compare_realimu(data, dataset_name='TotalCapture (Official Calibration)')

    data = torch.load('data/test_datasets/totalcapture_dipcalib.pt')
    compare_realimu(data, dataset_name='TotalCapture (DIP Calibration)', evaluate_tran=True)

    data = torch.load('data/test_datasets/dipimu.pt')
    compare_realimu(data, dataset_name='DIP_IMU')
