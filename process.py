import os
import pickle
import torch
import glob
import articulate as art
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
from scipy.interpolate import interp1d


def process_totalcapture(debug=False):
    print('======================== Processing TotalCapture Dataset ========================')
    joint_names = ['L_LowArm', 'R_LowArm', 'L_LowLeg', 'R_LowLeg', 'Head', 'Pelvis']
    vicon_gt_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/official'     # download from TotalCapture page
    imu_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/gryo_mag'          # download from TotalCapture page
    calib_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/imu'             # download from TotalCapture page
    DIP_smpl_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/DIP_smpl'     # SMPL pose calculated by DIP. Download from DIP page
    AMASS_smpl_dir = 'C:/yxy/datasets/TotalCapture/dataset_raw/AMASS_smpl' # SMPL pose calculated by AMASS. Download from AMASS page
    data = {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [], 'tran': [], 'AMASS_pose': [], 'DIP_pose': []}
    n_extracted_imus = len(joint_names)

    for subject_name in ['s1', 's2', 's3', 's4', 's5']:
        for action_name in sorted(os.listdir(os.path.join(imu_dir, subject_name))):
            # read imu file
            f = open(os.path.join(imu_dir, subject_name, action_name), 'r')
            line = f.readline().split('\t')
            n_sensors, n_frames = int(line[0]), int(line[1])
            R = torch.zeros(n_frames, n_extracted_imus, 4)
            a = torch.zeros(n_frames, n_extracted_imus, 3)
            w = torch.zeros(n_frames, n_extracted_imus, 3)
            m = torch.zeros(n_frames, n_extracted_imus, 3)
            for i in range(n_frames):
                assert int(f.readline()) == i + 1, 'parse imu file error'
                for _ in range(n_sensors):
                    line = f.readline().split('\t')
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        R[i, j] = torch.tensor([float(_) for _ in line[1:5]])  # wxyz
                        a[i, j] = torch.tensor([float(_) for _ in line[5:8]])
                        w[i, j] = torch.tensor([float(_) for _ in line[8:11]])
                        m[i, j] = torch.tensor([float(_) for _ in line[11:14]])
            R = art.math.quaternion_to_rotation_matrix(R).view(-1, n_extracted_imus, 3, 3)

            # read calibration file
            name = subject_name + '_' + action_name.split('_')[0].lower()
            RSB = torch.zeros(n_extracted_imus, 3, 3)
            RIM = torch.zeros(n_extracted_imus, 3, 3)
            with open(os.path.join(calib_dir, subject_name, name + '_calib_imu_bone.txt'), 'r') as f:
                n_sensors = int(f.readline())
                for _ in range(n_sensors):
                    line = f.readline().split()
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])  # wxyz
                        RSB[j] = art.math.quaternion_to_rotation_matrix(q)[0].t()
            with open(os.path.join(calib_dir, subject_name, name + '_calib_imu_ref.txt'), 'r') as f:
                n_sensors = int(f.readline())
                for _ in range(n_sensors):
                    line = f.readline().split()
                    if line[0] in joint_names:
                        j = joint_names.index(line[0])
                        q = torch.tensor([float(line[4]), float(line[1]), float(line[2]), float(line[3])])  # wxyz
                        RIM[j] = art.math.quaternion_to_rotation_matrix(q)[0].t()
            RSB = RSB.matmul(torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0.]]))  # change bone frame to SMPL
            RIM = RIM.matmul(torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1.]]))   # change global frame to SMPL

            # read root translation
            tran = []
            with open(os.path.join(vicon_gt_dir, subject_name.upper(), action_name.split('_')[0].lower(), 'gt_skel_gbl_pos.txt')) as f:
                idx = f.readline().split('\t').index('Hips')
                while True:
                    line = f.readline()
                    if line == '':
                        break
                    t = [float(_) * 0.0254 for _ in line.split('\t')[idx].split(' ')]   # inches_to_meters
                    tran.append([-t[0], t[1], -t[2]])
            tran = torch.tensor(tran)

            # read SMPL pose parameters calculated by AMASS
            f = os.path.join(AMASS_smpl_dir, subject_name, action_name.split('_')[0].lower() + '_poses.npz')
            AMASS_pose = None
            if os.path.exists(f):
                d = np.load(f)
                AMASS_pose = torch.from_numpy(d['poses'])[:, :72].float()
                root_rot = art.math.axis_angle_to_rotation_matrix(AMASS_pose[:, :3])
                root_rot = torch.tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]).matmul(root_rot)  # align global frame
                root_rot = art.math.rotation_matrix_to_axis_angle(root_rot)
                AMASS_pose[:, :3] = root_rot
                AMASS_pose[:, 66:] = 0  # hand

            # read SMPL pose parameters calculated by DIP
            f = os.path.join(DIP_smpl_dir, name + '.pkl')
            DIP_pose = None
            if os.path.exists(f):
                d = pickle.load(open(f, 'rb'), encoding='latin1')
                DIP_pose = torch.from_numpy(d['gt']).float()

            # align data
            n_aligned_frames = min(n_frames, tran.shape[0], AMASS_pose.shape[0] if AMASS_pose is not None else 1e8, DIP_pose.shape[0] if DIP_pose is not None else 1e8)
            if AMASS_pose is not None:
                AMASS_pose = AMASS_pose[-n_aligned_frames:]
            if DIP_pose is not None:
                DIP_pose = DIP_pose[-n_aligned_frames:]
            tran = tran[-n_aligned_frames:] - tran[-n_aligned_frames]
            R = R[-n_aligned_frames:]
            a = a[-n_aligned_frames:]
            w = w[-n_aligned_frames:]
            m = m[-n_aligned_frames:]

            # validate data (for debug purpose)
            if debug and DIP_pose is not None:
                model = art.ParametricModel('models/SMPL_male.pkl')
                DIP_pose = art.math.axis_angle_to_rotation_matrix(DIP_pose).view(-1, 24, 3, 3)
                syn_RMB = model.forward_kinematics_R(DIP_pose)[:, [18, 19, 4, 5, 15, 0]]
                real_RMB = RIM.transpose(1, 2).matmul(R).matmul(RSB)
                real_aM = RIM.transpose(1, 2).matmul(R).matmul(a.unsqueeze(-1)).squeeze(-1)
                print('real-syn imu ori err:', art.math.radian_to_degree(art.math.angle_between(real_RMB, syn_RMB).mean()))
                print('mean acc in M:', real_aM.mean(dim=(0, 1)))   # (0, +g, 0)

            # save results
            data['name'].append(name)
            data['RIM'].append(RIM)
            data['RSB'].append(RSB)
            data['RIS'].append(R)
            data['aS'].append(a)
            data['wS'].append(w)
            data['mS'].append(m)
            data['tran'].append(tran)
            data['AMASS_pose'].append(AMASS_pose)
            data['DIP_pose'].append(DIP_pose)
            print('Finish Processing %s' % name, '(no AMASS pose)' if AMASS_pose is None else '', '(no DIP pose)' if DIP_pose is None else '')

    os.makedirs('data/dataset_work/TotalCapture', exist_ok=True)
    torch.save(data, 'data/dataset_work/TotalCapture/data.pt')


def process_dipimu():
    print('======================== Processing DIP_IMU Dataset ========================')
    data_dir = 'C:/yxy/datasets/DIP_IMU/dataset_raw'
    imu_mask = [7, 8, 11, 12, 0, 2]  # head, spine2, belly, lchest, rchest, lshoulder, rshoulder, lelbow, relbow, lhip, rhip, lknee, rknee, lwrist, lwrist, lankle, rankle
    subject_names = ['s_%02d' % i for i in range(1, 11)]
    data = {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [], 'tran': [], 'pose': []}
    g = torch.tensor([0, -9.798, 0])
    for subject_name in subject_names:
        for motion_name in os.listdir(os.path.join(data_dir, subject_name)):
            f = os.path.join(data_dir, subject_name, motion_name)
            d = pickle.load(open(f, 'rb'), encoding='latin1')
            acc = torch.from_numpy(d['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(d['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(d['gt']).float()

            # fill nan with linear interpolation
            for i in range(ori.shape[0]):
                for j in range(6):
                    if torch.isnan(ori[i, j]).sum() > 0:
                        k1, k2 = i - 1, i + 1
                        while k1 >= 0 and torch.isnan(ori[k1, j]).sum() > 0: k1 -= 1
                        while k2 < ori.shape[0] and torch.isnan(ori[k2, j]).sum() > 0: k2 += 1
                        if k1 >= 0 and k2 < ori.shape[0]:
                            slerp = Slerp([k1, k2], Rotation.from_matrix(ori[[k1, k2], j].numpy()))
                            ori[k1 + 1:k2, j] = torch.from_numpy(slerp(list(range(k1 + 1, k2))).as_matrix()).float()
                        elif k1 < 0:
                            ori[:k2, j] = ori[k2, j]
                        elif k2 >= ori.shape[0]:
                            ori[k1 + 1:, j] = ori[k1, j]
                    if torch.isnan(acc[i, j]).sum() > 0:
                        k1, k2 = i - 1, i + 1
                        while k1 >= 0 and torch.isnan(acc[k1, j]).sum() > 0: k1 -= 1
                        while k2 < ori.shape[0] and torch.isnan(acc[k2, j]).sum() > 0: k2 += 1
                        if k1 >= 0 and k2 < ori.shape[0]:
                            lerp = interp1d([k1, k2], acc[[k1, k2], j].numpy(), axis=0)
                            acc[k1 + 1:k2, j] = torch.from_numpy(lerp(list(range(k1 + 1, k2)))).float()
                        elif k1 < 0:
                            acc[:k2, j] = acc[k2, j]
                        elif k2 >= ori.shape[0]:
                            acc[k1 + 1:, j] = acc[k1, j]

            if torch.isnan(acc).sum() > 0 or torch.isnan(ori).sum() > 0 or torch.isnan(pose).sum() > 0:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))
                continue

            # synthesize wS and mS, calculate aS
            w = art.math.rotation_matrix_to_axis_angle(ori[:-1].transpose(2, 3).matmul(ori[1:])).view(-1, ori.shape[1], 3) * 60
            w = torch.cat((w, torch.zeros_like(w[:1])))
            m = ori.transpose(2, 3).matmul(torch.tensor([1, 0, 0.]).unsqueeze(-1)).squeeze(-1)
            a = ori.transpose(2, 3).matmul((acc - g).unsqueeze(-1)).squeeze(-1)

            name = subject_name.replace('_', '') + '_' + motion_name[:-4]
            data['name'].append(name)
            data['RIM'].append(torch.eye(3).repeat(6, 1, 1))
            data['RSB'].append(torch.eye(3).repeat(6, 1, 1))
            data['RIS'].append(ori)
            data['aS'].append(a)
            data['wS'].append(w)
            data['mS'].append(m)
            data['tran'].append(torch.zeros(pose.shape[0], 3))
            data['pose'].append(pose)
            print('Finish Processing %s' % name)

    os.makedirs('data/dataset_work/DIP_IMU', exist_ok=True)
    torch.save(data, 'data/dataset_work/DIP_IMU/data.pt')


def process_amass():
    print('======================== Processing AMASS Dataset ========================')
    data_dir = 'D:/Admin/Data/AMASS/dataset_raw'
    names = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU', 'DFaust67',
             'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD', 'BioMotionLab_NTroje',
             'BMLhandball', 'MPI_Limits', 'TotalCapture']   # align with previous works
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    data = {'name': [], 'RIM': [], 'RSB': [], 'RIS': [], 'aS': [], 'wS': [], 'mS': [], 'tran': [], 'pose': []}
    for name in names:
        print('Processing %s' % name)
        for npz_fname in glob.glob(os.path.join(data_dir, name, name, '*/*_poses.npz')) + glob.glob(os.path.join(data_dir, name, name, '*/*_stageii.npz')):
            seq_name = npz_fname[npz_fname.rfind(name):-4]
            try:
                cdata = np.load(npz_fname, allow_pickle=True)
                if 'mocap_framerate' in cdata:
                    framerate = int(cdata['mocap_framerate'])
                elif 'mocap_frame_rate' in cdata:
                    framerate = int(cdata['mocap_frame_rate'])
                else:
                    print('\tFail to process %s: no framerate' % seq_name)
                    continue
                if cdata['poses'].shape[0] < framerate * 0.5:
                    print('\tFail to process %s: too short' % seq_name)
                    continue
                if framerate == 120:
                    pose = torch.from_numpy(cdata['poses'][::2].astype(np.float32)).view(-1, 156)[:, :72]
                    tran = torch.from_numpy(cdata['trans'][::2].astype(np.float32)).view(-1, 3)
                elif framerate == 60 or framerate == 59:
                    pose = torch.from_numpy(cdata['poses'].astype(np.float32)).view(-1, 156)[:, :72]
                    tran = torch.from_numpy(cdata['trans'].astype(np.float32)).view(-1, 3)
                else:
                    origin_pose = cdata['poses'].reshape(-1, 52, 3)
                    origin_tran = cdata['trans'].reshape(-1, 3)
                    origin_t = np.arange(origin_pose.shape[0]) / framerate
                    t = np.arange(0, origin_t[-1], 1 / 60)
                    pose = np.empty((len(t), 24, 3))
                    for i in range(24):
                        pose[:, i] = Slerp(origin_t, Rotation.from_rotvec(origin_pose[:, i]))(t).as_rotvec()
                    tran = interp1d(origin_t, origin_tran, axis=0)(t)
                    pose = torch.from_numpy(pose.astype(np.float32)).view(-1, 72)
                    tran = torch.from_numpy(tran.astype(np.float32)).view(-1, 3)
            except Exception as e:
                print('\tFail to process %s:' % seq_name, e)
            pose[:, :3] = art.math.rotation_matrix_to_axis_angle(amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, :3])))
            tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
            data['name'].append(seq_name)
            data['pose'].append(pose.clone())
            data['tran'].append(tran.clone())
            print('\tFinish Processing %s: n_frames %d' % (seq_name, pose.shape[0]))

    assert len(data['name']) > 0, 'cannot find AMASS dataset'
    os.makedirs('data/dataset_work/AMASS', exist_ok=True)
    torch.save(data, 'data/dataset_work/AMASS/data.pt')


if __name__ == '__main__':
    # import sys; sys.stdout = open('data/dataset_work/log.txt', 'a')
    process_amass()
    process_totalcapture()
    process_dipimu()
