'''
render double dance in mesh
'''
import os
from matplotlib import axis
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import subprocess
import sys, glob
import cv2
from smplx import SMPL, SMPLH, SMPLX
from tqdm import tqdm
import torch, pickle
from data.utils.motion_process import recover_from_ric, recover_from_ric266, recover_from_ric266v, recover_from_smplx_v, recover_from_smplx, recover_from_smplx_globalinit_v, recover_from_smplx_globalinit_x_grad
import numpy as np
# from data.utils.smplfk import SMPLX_Skeleton, do_smplxfk
import argparse
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
import pyrender
import trimesh


def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat
def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat
def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat
def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax
def ax_from_qua(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax

def motion_data_load_process(motionfile):
    setbetas = None
    if motionfile.split(".")[-1] == "pkl":
        pkl_data = pickle.load(open(motionfile, "rb"))
        if "smpl_poses" in pkl_data.keys():
            smpl_poses = pkl_data["smpl_poses"]
            modata = np.concatenate((pkl_data["smpl_trans"], smpl_poses), axis=1)
        elif "trans" in pkl_data.keys():
            print(pkl_data.keys())
            print('trans', pkl_data["trans"].shape)
            print('smpl_poses', pkl_data["pose"].shape)
            print('beta', pkl_data['beta'].shape)

            smpl_poses = pkl_data["pose"]
            if smpl_poses.shape[1] != 165:
                if smpl_poses.shape[1]==55 and smpl_poses.shape[2]==3:
                    smpl_poses = smpl_poses.reshape(smpl_poses.shape[0], 165)
            modata = np.concatenate((pkl_data["trans"], smpl_poses), axis=1)
        if modata.shape[1] == 69:
            hand_zeros = np.zeros([modata.shape[0], 90], dtype=np.float32)
            modata = np.concatenate((modata, hand_zeros), axis=1)
        print(modata.shape)
        # modata[:, 1] = modata[:, 1] + 1
        setbetas = pkl_data['beta']

        return modata, setbetas

    elif motionfile.split(".")[-1] == "npy":
        modata = np.load(motionfile)
        if len(modata.shape) == 3 and modata.shape[1]%8==0:
            print("modata has 3 dim , reshape the batch to time!!!")
            modata = modata.reshape(-1, modata.shape[-1])
        if modata.shape[-1] == 315:
            print("modata.shape is:", modata.shape)
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 319:
            print("modata.shape is:", modata.shape)
            modata = modata[:,4:]
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 159:
            print("modata.shape is:", modata.shape)
            # rot6d = torch.from_numpy(modata[:,3:])
            # T,C = rot6d.shape
            # rot6d = rot6d.reshape(-1,6)
            # axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            # modata = np.concatenate((modata[:,:3], axis), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 168:
            print("modata.shape is:", modata.shape)
            modata = np.concatenate([modata[:,:3+22*3],  modata[:,-90:]], axis=-1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 135:
            print("modata.shape is:", modata.shape)
            if len(modata.shape) == 3 and modata.shape[0] ==1:
                modata = modata.squeeze(0)
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 139:
            print("modata.shape is:", modata.shape)
            modata = modata[:,4:]
            rot6d = torch.from_numpy(modata[:,3:])
            T,C = rot6d.shape
            rot6d = rot6d.reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((modata[:,:3], axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 268 or modata.shape[-1] == 271:
            print("modata.shape is:", modata.shape)
            T, _ = modata.shape
            modata = torch.from_numpy(modata).float()
            r_pos = torch.zeros([T, 3])   #.to(modata.device)
            r_pos[1:, [0, 2]] = modata[:-1, [0, 2]]
            r_pos = torch.cumsum(r_pos, dim=-2)
            r_pos[..., 1] = modata[..., 1]
            
            rot6d = modata[:,3:3+22*6].reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,-1).detach().cpu().numpy()
            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((r_pos, axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 144:
            print("attention! there are two cases of 144!!!")
            
            print("modata.shape is:", modata.shape)
            T, _ = modata.shape
            modata = torch.from_numpy(modata).float()
            r_pos = modata[:,:3]   #.to(modata.device)
            rot6d = modata[:,6:6+22*6].reshape(-1,6)
            axis = ax_from_6v(rot6d).view(T,22*3).detach().cpu().numpy()

            hand_zeros = torch.zeros([T, 90]).to(rot6d).detach().cpu().numpy()
            modata = np.concatenate((r_pos, axis, hand_zeros), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 211:
            print("modata.shape is:", modata.shape)
            T, _ = modata.shape
            modata = torch.from_numpy(modata).float()
            r_pos = modata[:,:3]   #.to(modata.device)
            qua = modata[:,3:].reshape(T,-1,4)
            axis = ax_from_qua(qua).view(T,52*3).detach().cpu().numpy()
            modata = np.concatenate((r_pos, axis), axis=1)
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 221:
            print("modata.shape is:", modata.shape)
            T, _ = modata.shape
            modata = torch.from_numpy(modata).float()
            r_pos = modata[:,:3]   #.to(modata.device)
            qua = modata[:,3:211].reshape(T,-1,4)
            setbetas = modata[:,211:]
            axis = ax_from_qua(qua).view(T,52*3).detach().cpu().numpy()
            modata = np.concatenate((r_pos, axis), axis=1)
            modata[:, 1] = modata[:, 1] - 1.3 + 0.44
            print("modata.shape is:", modata.shape)
        elif modata.shape[-1] == 233:
            print("modata.shape is:", modata.shape)
            T, _ = modata.shape
            modata = torch.from_numpy(modata).float()
            r_pos = modata[:,:3]   #.to(modata.device)
            qua = modata[:,3:223].reshape(T,-1,4)
            setbetas = modata[0,223:233].detach().cpu().numpy()
            axis = ax_from_qua(qua).view(T,55*3).detach().cpu().numpy()
            modata = np.concatenate((r_pos, axis), axis=1)
            # modata[:, 1] = modata[:, 1]
            print("modata.shape is:", modata.shape)
        else:
            print("modata.shape is:", modata.shape)
            raise("shape error!")
            
        # modata[:, 1] = modata[:, 1]# + 1.3
        if setbetas is None:
            setbetas = torch.zeros([modata.shape[0], 10])
            setbetas = setbetas.detach().cpu().numpy()
     
        return modata, setbetas

def get_joints_from_file(file,use_rela=False):
    rela_dis=None
    if file[-3:] == 'pkl':
        pkl_data = pickle.load(open(file, 'rb'))
        # file_li.append(os.path.join(outdir, file.replace('pkl', 'gif')))
        smpl_poses = torch.from_numpy(pkl_data["smpl_poses"]).reshape(-1, 22, 3)
        smpl_trans = torch.from_numpy(pkl_data["smpl_trans"])
        smpl_poses = ax_to_6v(smpl_poses).view(smpl_trans.shape[0], 132)
        data139 = torch.cat([torch.zeros(smpl_poses.shape[0], 4).to(smpl_poses), smpl_trans, smpl_poses], dim=-1)
        print("data139", data139.shape)
        joints = do_smplxfk(data139, smplx)[:, :22, :]
    elif file[-3:] == 'npy':
        motion = np.load(file)  # [30:150]
        print('motion type', type(motion))
        # file_li.append(os.path.join(outdir, file.replace('npy', 'gif')))
        print("motion shape", motion.shape)
        if motion.shape[1] == 263 or motion.shape[1] == 266:
            # motion = (motion * std) + mean
            motion = torch.from_numpy(motion)
        elif len(motion.shape) == 3 and motion.shape[-1] == 263:
            motion = torch.from_numpy(motion)[8]
        if motion.shape[1] == 263:
            joints = recover_from_ric(motion.to(dtype=torch.float), joints_num)  # .detach().cpu().numpy()
        elif motion.shape[1] == 266:
            joints = recover_from_ric266(motion.to(dtype=torch.float), joints_num)
            print("in 266")
            print("joints  11", joints[0,:3])
            print("joints  11", joints.shape)
        elif motion.shape[1] == 338:
            motion = torch.from_numpy(motion)
            # joints = recover_from_smplx_v(motion[:,:338].to(dtype=torch.float), joints_num)
            joints = recover_from_smplx_globalinit_v(motion[:,:338].to(dtype=torch.float), joints_num)
            print('joints', joints.shape)
        elif motion.shape[1] == 340:
            motion = torch.from_numpy(motion)
            joints = recover_from_smplx_v(motion[:, :338].to(dtype=torch.float), joints_num)
            rela_dis = np.pad(motion[:,338:], ((0, 0), (0, 1)), mode='constant', constant_values=0)
        elif motion.shape[1] > 338:
            if motion.shape[-1] == 4978:
                print('motion shape1······················', motion.shape)
                label = motion[:, -710:]
                motion_  = motion[:, :-710]
                joints_num = 710
            else:
                label = None
                motion_ = motion
            print('motion_ shape1······················', motion_.shape)
            motion_ = torch.from_numpy(motion_)
            joints = recover_from_smplx(motion_.to(dtype=torch.float), joints_num=710)
            print('joints', joints.shape)
        elif motion.shape[2] == 4978:
            motion = motion[0]
            print('motion shape1······················', motion.shape)
            label = motion[:, -710:]
            motion_  = motion[:, :-710]
            joints_num = 710
            print('motion_ shape1······················', motion_.shape)
            motion_ = torch.from_numpy(motion_)
            joints = recover_from_smplx(motion_.to(dtype=torch.float), joints_num=710)
            print('joints', joints.shape)
        else:
            print("motion shape", motion.shape)
            raise("error of motion shape")

    # joints = joints.reshape(joints.shape[0], joints_num * 3).detach().cpu().numpy()
    # roott = joints[:1, :3]  # the root Tx72 (Tx(24x3))
    # joints = joints - np.tile(roott, (1, joints_num))
    joints = joints.reshape(-1, joints_num, 3)
    if use_rela:
        joints = np.concatenate((joints, rela_dis[:, np.newaxis, :]), axis=1)  # T,56,3


    return joints, label

def look_at(eye, center, up):
    front = eye - center
    front = front / np.linalg.norm(front)
    right = np.cross(up, front)
    right = right / np.linalg.norm(right)
    up_new = np.cross(front, right)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.stack([right, up_new, front]).transpose()
    camera_pose[:3, 3] = eye
    return camera_pose

def create_checkerboard_texture(size=2, num_tiles=8):
    # Create a checkerboard pattern
    tile_size = size / num_tiles
    checkerboard = np.zeros((num_tiles, num_tiles, 3), dtype=np.uint8)
    for i in range(num_tiles):
        for j in range(num_tiles):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = [255, 255, 255]  # White color
            else:
                checkerboard[i, j] = [127, 127, 127]  # Gray color

    # Repeat the pattern to cover the entire texture
    checkerboard = np.kron(checkerboard, np.ones((tile_size, tile_size, 1), dtype=np.uint8))
    return checkerboard


class MovieMaker():
    def __init__(self, save_path) -> None:

        self.mag = 2
        self.eyes = np.array([[3, -3, 2], [0, 0, -2], [0, 0, 4], [0, 2.5, 4.5], [0, 2, 4], [0, 3, 5]])       
        self.centers = np.array([[0, 0, 0], [0, 0, 0], [0, 0.5, 0], [0, 0.5, 0], [0, 0.5, 0], [0, 0.5, 0]])
        self.ups = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.save_path = save_path

        self.fps = args.fps
        self.img_size = (1200, 1200)

        SMPLH_path = "/data2/smpl_model/smplh/SMPLH_MALE.pkl"
        SMPL_path = "/data2/smpl_model/smpl/SMPL_MALE.pkl"
        SMPLX_path = "/data2/smpl_model/smplx/SMPLX_NEUTRAL.npz"
        trimesh_path = '/data2/floor/NORMAL_new.obj'
        self.faces655 = np.load('/data2/dataset/RepairedDouble/beforesplit/vertices/faces655.npy')

        self.smplh = SMPLH(SMPLH_path, use_pca=False, flat_hand_mean=False)
        self.smplh.to(f'cuda:{args.gpu}').eval()

        self.smpl = SMPL(SMPL_path)
        self.smpl.to(f'cuda:{args.gpu}').eval()

        self.smplx = SMPLX(SMPLX_path, use_pca=False, flat_hand_mean=False).eval()
        self.smplx.to(f'cuda:{args.gpu}').eval()
        self.scene = pyrender.Scene()


        self.mesh = trimesh.load(trimesh_path)
        floor_mesh = pyrender.Mesh.from_trimesh(self.mesh)
        self.scene.add(floor_mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = look_at(self.eyes[5], self.centers[5], self.ups[5])  # 2
        self.scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3)
        self.scene.add(light, pose=camera_pose)
        self.r = pyrender.OffscreenRenderer(self.img_size[0], self.img_size[1])

    def save_video(self, save_path, color_list):
        # save_path = os.path.join(save_path,'move.mp4')
        f = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(save_path, f, self.fps, self.img_size)
        for i in range(len(color_list)):
            videowriter.write(color_list[i][:, :, ::-1])
        videowriter.release()
    def save_two_video(self, save_path, color_list0,color_list1):
        # save_path = os.path.join(save_path,'move.mp4')
        f = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(save_path, f, self.fps, self.img_size)
        for i in range(len(color_list0)):
            videowriter.write(color_list0[i][:, :, ::-1])
        for i in range(len(color_list1)):
            videowriter.write(color_list1[i][:, :, ::-1])
        videowriter.release()
    def get_imgs(self, motion):
        meshes = self.motion2mesh(motion)
        imgs = self.render_imgs(meshes)
        return np.concatenate(imgs, axis=1)

    def motion2mesh(self, motion):
        if args.mode == "smpl":
            output = self.smpl.forward(
                betas=torch.zeros([motion.shape[0], 10]).to(motion.device),
                transl=motion[:, :3],
                global_orient=motion[:, 3:6],
                body_pose=torch.cat([motion[:, 6:69], motion[:, 69:72], motion[:, 114:117]], dim=1)
            )
        elif args.mode == "smplh":
            output = self.smplh.forward(
                betas=torch.zeros([motion.shape[0], 10]).to(motion.device),
                # transl = motion[:,:3],
                transl=torch.tensor([[0, 0, -1]]).expand(motion.shape[0], -1).to(motion.device),
                global_orient=motion[:, 3:6],
                body_pose=motion[:, 6:69],
                left_hand_pose=motion[:, 69:114],
                right_hand_pose=motion[:, 114:159],
            )
        elif args.mode == "smplx":
            setbetas = torch.zeros()([motion.shape[0], 10]).to(motion.device)
            output = self.smplx.forward(
                betas=setbetas,
                # transl = motion[:,:3],
                transl=motion[:, :3],
                global_orient=motion[:, 3:6],
                body_pose=motion[:, 6:69],
                jaw_pose=torch.zeros([motion.shape[0], 3]).to(motion),
                leye_pose=torch.zeros([motion.shape[0], 3]).to(motion),
                reye_pose=torch.zeros([motion.shape[0], 3]).to(motion),
                left_hand_pose = motion[:,69:69+15*3],
                right_hand_pose = motion[:,-45:],
                expression=torch.zeros([motion.shape[0], 10]).to(motion),
            )
        # tespo
        meshes = []
        for i in range(output.vertices.shape[0]):
            if args.mode == 'smplh':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
            elif args.mode == 'smplx':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
            elif args.mode == 'smpl':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
            # mesh.export(os.path.join(self.save_path, f'{i}.obj'))
            meshes.append(mesh)

        return meshes

    def render_multi_view(self, meshes, music_file, tab='', eyes=None, centers=None, ups=None, views=1):
        if eyes and centers and ups:
            assert eyes.shape == centers.shape == ups.shape
        else:
            eyes = self.eyes
            centers = self.centers
            ups = self.ups

        for i in range(views):
            color_list = self.render_single_view(meshes, eyes[1], centers[1], ups[1])
            movie_file = os.path.join(self.save_path, tab + '-' + str(i) + '.mp4')
            output_file = os.path.join(self.save_path, tab + '-' + str(i) + '-music.mp4')
            self.save_video(movie_file, color_list)
            if music_file is not None:
                subprocess.run(
                    ['/home/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, '-i', music_file,
                     '-shortest', output_file])
            else:
                subprocess.run(['/home/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, output_file])
            os.remove(movie_file)
    def render_two_person(self,meshes0,meshes1, label0, label1):
        # print("frames",len(meshes0),len(meshes0[0]))
        num = min(len(meshes0),len(meshes1))
        meshes0 = meshes0[:num]
        print('--meshes0--', len(meshes0))
        meshes1 = meshes1[:num]
        print('--meshes1--', len(meshes1))
        # red_color =  [255,192,203,1.000]   # [1.0, 0.0, 0.0, 1.0]  
        # red_color = [x/255 for x in red_color]
        # material_red = pyrender.MetallicRoughnessMaterial(baseColorFactor=red_color)
        green_color =  [127,255,212,1.000] #[127,255,212,1.000]     # RGBA
        green_color = [x/255 for x in green_color]
        material_green = pyrender.MetallicRoughnessMaterial(baseColorFactor=green_color)

        blue_color = [	30, 144, 255, 1.000]# [60, 140, 196, 1.0] 
        blue_color = [x/255 for x in blue_color]
        material_blue = pyrender.MetallicRoughnessMaterial(baseColorFactor=blue_color)
        color_list = []
        for i in tqdm(range(num)):
            mesh_nodes = []
            for idx, mesh in enumerate(meshes0[i]):
                # if label0[i][idx] > 0:
                #     render_mesh = pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0, 0.0, 0.0, 1.0]))
                # else:
                render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material_green)     
                # render_mesh = pyrender.Mesh.from_trimesh(mesh)                              
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            for mesh in meshes1[i]:
                # if label1[i][idx] > 1:
                #     render_mesh = pyrender.Mesh.from_trimesh(mesh, material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 1.0, 1.0]))
                # else:
                render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material_blue)
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            color = color.copy()
            color_list.append(color)
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)
        return color_list
    def render_single_view(self, meshes):
        num = len(meshes)
        color_list = []
        for i in tqdm(range(num)):
            mesh_nodes = []
            for mesh in meshes[i]:
                render_mesh = pyrender.Mesh.from_trimesh(mesh)
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            color = color.copy()
            color_list.append(color)
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)
        return color_list

    def render_imgs(self, meshes):
        colors = []
        for mesh in meshes:
            render_mesh = pyrender.Mesh.from_trimesh(mesh)
            mesh_node = self.scene.add(render_mesh)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            colors.append(color)
            self.scene.remove_node(mesh_node)

        return colors
        # cv2.imwrite(os.path.join(self.save_path, 'test.jpg'), color[:,:,::-1])

    def run_two(self,seq_rot0, seq_rot1, setbetas0, setbetas1, label0, label1, music_file=None, dance_name='000',tab0='',tab1='', save_pt=False):
        meshes0, lable_0 = self.get_out_from_seq(seq_rot0, setbetas0, label0, tab=tab0,save_pt=save_pt)
        meshes1, lable_1 = self.get_out_from_seq(seq_rot1, setbetas1, label1, tab=tab1, save_pt=save_pt)
        color_list = self.render_two_person(meshes0,meshes1, lable_0, lable_1)
        movie_file = os.path.join(self.save_path, dance_name + 'tmp.mp4')
        output_file = os.path.join(self.save_path, dance_name + 'z.mp4')
        self.save_video(movie_file, color_list)
        if music_file is not None:
            subprocess.run(
                ['/home/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, '-i', music_file,
                 '-shortest',
                 output_file])
        else:
            subprocess.run(
                ['/home/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, output_file])
        os.remove(movie_file)

    def get_out_from_seq(self, seq_rot, setbetas, label, tab='', save_pt=False):
        if isinstance(seq_rot, np.ndarray):
            seq_rot = torch.tensor(seq_rot, dtype=torch.float32, device=f'cuda:{args.gpu}')
        if save_pt:
            torch.save(seq_rot.detach().cpu(), os.path.join(self.save_path, tab + '_pose.pt'))

        if len(seq_rot.shape) == 2 and args.mode[:4] == 'smpl':
            B, D = seq_rot.shape
            if setbetas is not None:
                setbetas = torch.from_numpy(setbetas).to(seq_rot)
            output=None
            if args.mode == "smpl":
                print("using smpl!!!")
                output = self.smpl.forward(
                    betas=setbetas.unsqueeze(0).repeat(seq_rot.shape[0], 1),
                    transl=seq_rot[:, :3],
                    global_orient=seq_rot[:, 3:6],
                    body_pose=torch.cat([seq_rot[:, 6:69], seq_rot[:, 69:72], seq_rot[:, 114:117]], dim=1)
                )

            elif args.mode == "smplh":
                print("using smplh!!!")
                output = self.smplh.forward(
                    betas=setbetas.unsqueeze(0).repeat(seq_rot.shape[0], 1),
                    transl=seq_rot[:, :3],
                    global_orient=seq_rot[:, 3:6],
                    body_pose=seq_rot[:, 6:69],
                    left_hand_pose=seq_rot[:, 69:114],
                    right_hand_pose=seq_rot[:, 114:],  # torch.zeros([seq_rot.shape[0], 45]).to(seq_rot.device),
                    expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                )

            elif args.mode == "smplx":
                output = self.smplx.forward(
                    betas=setbetas.unsqueeze(0).repeat(seq_rot.shape[0], 1),
                    transl=seq_rot[:, :3],
                    global_orient=seq_rot[:, 3:6],
                    body_pose=seq_rot[:, 6:69],
                    jaw_pose=seq_rot[:, 69:72],
                    leye_pose=seq_rot[:, 72:75],
                    reye_pose=seq_rot[:, 75:78],
                    left_hand_pose = seq_rot[:,78:78+45],
                    right_hand_pose = seq_rot[:,-45:],
                    expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot),
                )
            N, V, DD = output.vertices.shape  # 150, 6890, 3


            # '''Put on Floor'''
            floor_height = output.joints[:,:22].detach().cpu().numpy().min(axis=0).min(axis=0)[1]
            print('floor_height', floor_height)
            vertices = output.vertices
            vertices[:, :, 1] -= floor_height
            vertices = vertices.reshape((B, -1, V, DD))  # # 150, 1, 6890, 3
        elif len(seq_rot.shape) == 3  and args.mode == 'ver655':       
            B, _, _ = seq_rot.shape
            vertices = seq_rot
            label = label[:,55:]

        meshes = []
        labels = []
        for i in range(B):
            view = []
            if args.mode == 'smplh':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
            elif args.mode == 'smplx':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
            elif args.mode == 'smpl':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
            elif args.mode == 'ver655':
                mesh = trimesh.Trimesh(vertices[i].cpu(), self.faces655)
            view.append(mesh)
            meshes.append(view)
            if args.mode == 'ver655':
                labels.append(label[i])
        return meshes, labels
    def run(self, seq_rot, music_file=None, tab='', save_pt=False):
        if isinstance(seq_rot, np.ndarray):
            seq_rot = torch.tensor(seq_rot, dtype=torch.float32, device=f'cuda:{args.gpu}')

        if save_pt:
            torch.save(seq_rot.detach().cpu(), os.path.join(self.save_path, tab + '_pose.pt'))

        B, D = seq_rot.shape
        if args.mode == "smpl":
            print("using smpl!!!")
            output = self.smpl.forward(
                betas=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl=seq_rot[:, :3],
                global_orient=seq_rot[:, 3:6],
                body_pose=torch.cat([seq_rot[:, 6:69], seq_rot[:, 69:72], seq_rot[:, 114:117]], dim=1)
            )

        elif args.mode == "smplh":
            print("using smplh!!!")
            output = self.smplh.forward(
                betas=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl=seq_rot[:, :3],
                global_orient=seq_rot[:, 3:6],
                body_pose=seq_rot[:, 6:69],
                left_hand_pose=seq_rot[:, 69:114],
                # torch.zeros([seq_rot.shape[0], 45]).to(seq_rot.device),      # seq_rot[:,69:114],
                right_hand_pose=seq_rot[:, 114:],  # torch.zeros([seq_rot.shape[0], 45]).to(seq_rot.device),      #
                expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
            )

        elif args.mode == "smplx":
            setbetas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device)
            setbetas[:, 1] = -1.2
            output = self.smplx.forward(
                betas=setbetas,
                # transl = motion[:,:3],
                transl=seq_rot[:, :3],
                global_orient=seq_rot[:, 3:6],
                body_pose=seq_rot[:, 6:69],
                jaw_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                leye_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                reye_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                left_hand_pose=torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
                right_hand_pose=torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
                expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot),
            )

        N, V, DD = output.vertices.shape  # 150, 6890, 3
        vertices = output.vertices.reshape((B, -1, V, DD))  # # 150, 1, 6890, 3

        meshes = []
        for i in range(B):
            view = []
            for v in vertices[i]:
                # vertices[:,2] *= -1
                if args.mode == 'smplh':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
                elif args.mode == 'smplx':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
                elif args.mode == 'smpl':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
                elif args.mode == 'ver655':
                    mesh = trimesh.Trimesh(vertices[i].cpu(), self.faces655)
                view.append(mesh)
            meshes.append(view)

        color_list = self.render_single_view(meshes)
        movie_file = os.path.join(self.save_path, tab + 'tmp.mp4')
        output_file = os.path.join(self.save_path, tab + 'z.mp4')
        self.save_video(movie_file, color_list)
        if music_file is not None:
            subprocess.run(
                ['/home/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, '-i', music_file, '-shortest',
                 output_file])
        else:
            subprocess.run(['/home/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, output_file])
        os.remove(movie_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--modir", type=str, default="/home/Documents/dataset/RepairedDouble/new_xyz_vecs_128init")
    parser.add_argument("--musdir", type=str, default="/home/Documents/dataset/RepairedDouble/musicfea_128")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--joints_num", type=int, default=710)   # 710
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="ver655", choices=['smpl', 'smplh', 'smplx', "ver655"])

    # parser.add_argument("--song", type=str, default=None)
    args = parser.parse_args()
    print(args.gpu)
    joints_num = args.joints_num

    device = f'cuda:{args.gpu}'
    modir = args.modir
    if args.outdir is not None:
        outdir = os.path.join(modir, args.outdir)
    else:
        outdir = os.path.join(modir, 'camera3')
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    faces = np.load('/data2/dataset/RepairedDouble/beforesplit/vertices/faces655.npy')
    colors = np.ones((655, 4), dtype=np.float32) * 0.8
    motion_gtdir = '/data2/dataset/RepairedDouble/after_split/30fps/smplxdata_128_30fps'

    for filef in os.listdir(modir):
        if filef.split('.')[-1] not in ['pkl', 'npy']:
            continue
        if filef.endswith('l0.npy') or filef.endswith('l0.pkl'):                             # leader is 0
            if filef.split('@')[0].split('_')[-1] != '1':          # if file is not leader, continue
                continue
            filel = filef.replace('_1@', '_0@')
        if filef.endswith('l1.npy') or filef.endswith('l1.pkl'): 
            if filef.split('@')[0].split('_')[-1] != '0':          # if is not leader, continue
                continue
            filel = filef.replace('_0@', '_1@')
        if filef.endswith('_0.pkl') or filef.endswith('_1.pkl'):   # render gT
            if filef.endswith('_0.pkl'):        # not follower, continue
                continue
            else:
                filel = filef.replace('_1.pkl', '_0.pkl')

        if filef[0] == 'M':
            continue

        flag = False
        for exists_file in os.listdir(outdir):
            if filef[:-5] in exists_file:
                flag = True
                break
            else:
                flag = False
        if flag:
            print("exist", filef)
            continue

        
        # if filef.split('.')[-1] == 'npy':
        if args.mode == "ver655":
            modata0, label0 = get_joints_from_file(os.path.join(motion_gtdir, filel), use_rela=False)
            modata0 = modata0[:, 55:]
            print('modata0', modata0.shape)
            modata1, label1 = get_joints_from_file(os.path.join(modir, filef), use_rela=False)
            modata1 = modata1[:, 55:]
            print('modata1', modata1.shape)
            setbetas0 = None
            setbetas1 = None
        elif args.mode == "smplx":
            modata0, setbetas0 = motion_data_load_process(os.path.join(motion_gtdir, filel[:-3] + 'pkl'))
            modata1, setbetas1 = motion_data_load_process(os.path.join(modir, filef))
            label0 = label1 = None
        
        mid = (modata0[0,[0,2]]+modata1[0,[0,2]])/2
        print('mid is' , mid)
        modata0[:,[0,2]] = modata0[:,[0,2]] - mid
        modata1[:,[0,2]] = modata1[:,[0,2]] - mid

        visualizer = MovieMaker(save_path=outdir)
        visualizer.run_two(modata0[:], modata1[:], setbetas0, setbetas1, label0, label1,
                           dance_name=os.path.basename(filef.split(".")[0]),
                           tab0=os.path.basename(filel).split(".")[0],
                           tab1=os.path.basename(filef).split(".")[0], 
                           music_file=None)
        
'''
render smplx file
--modir 'smplx pkl path' --mode smplx --fps 8 --outdir ./video_gt
render ver655 file
--modir 'ver655 npy path' --mode ver655 --fps 8
'''

