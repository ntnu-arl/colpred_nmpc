import numpy as np
from collision_predictor_mpc import COLPREDMPC_CONFIG_DIR, config, controller, ros_wrapper
from collision_predictor_mpc.utils import quat2euler, euler2quat, quat2rot
import time
import os
import rospy
import argparse


if __name__ == '__main__':
    ## args
    parser = argparse.ArgumentParser(description='Main controller loop with ros interface.')
    parser.add_argument('--cfg', dest='cfg_file', default='ros_params.yaml', type=str) help='Config file to be used.')
    args = parser.parse_args()

    ## nmpc
    mission_config = os.path.join(COLPREDMPC_CONFIG_DIR, args.cfg_file)
    cfg = config.Config(mission_config)
    nmpc = controller.NMPC(cfg)

    ## ROS
    system = ros_wrapper.RosWrapper(cfg)

    ## reference
    p_hover = None
    p_des = None
    p_des_body = np.array(cfg.mission['p_des'])
    zref = cfg.mission['zref']
    vref = cfg.mission['vref']

    ## logs
    logfile = cfg.mission['logfile']
    hlogfile = cfg.mission['horizon_logfile']
    with open(logfile, 'w') as flog:
        log_header = np.array(['ts', 'cpt', 'flag', 'obj', 'x', 'y', 'z', 'qw', 'qz', 'vx', 'ax', 'vz', 'wz', 'colpred', 'z_r', 'Wvx_r', 'Wvy_r']).reshape(1,-1)
        np.savetxt(flog, log_header, fmt='%s', delimiter=' ', newline='\n')
    with open(hlogfile, 'w') as flog:
        log_header = np.array(['ts', 'i', 'x', 'y', 'z', 'qw', 'qz', 'vx', 'colpred']).reshape(1,-1)
        np.savetxt(flog, log_header, fmt='%s', delimiter=' ', newline='\n')

    ## init network
    nmpc.set_new_image(np.ones((480,270), dtype=np.float32), [0,0,0], [0,0,0])
    nmpc.set_colpred_flag(1)
    nmpc.solve()
    nmpc.set_colpred_flag(0)

    ## loop
    ts = cfg.mission['control_loop_time']*1e-3
    print('entering nmpc loop')
    while not rospy.is_shutdown():
        ## while on start, send 0 vel cmd
        if not system.start:
            nmpc.x0 = None
            system.send_commands([0,0,0,0])
            rospy.sleep(ts)
            continue
        tic = time.time()

        ## poll for new image
        system_img = system.get_image()
        if system_img is not None:
            img = system_img[0]
            system.send_img(img)
            img_p = system_img[1]
            img_q = system_img[2]
            nmpc.set_new_image(img, img_p, np.array(quat2euler(img_q)).flatten())

        ## wait for new state to come in
        system_state = system.get_state()
        if system_state is None:
            time.sleep(1e-3)
            continue

        p = system_state[0]
        q = system_state[1]
        v = system_state[2]
        yaw = quat2euler(q)[2]
        qyaw = np.array(euler2quat([0, 0, yaw])).flatten()
        x0 = np.array([p[0], p[1], p[2], qyaw[0], qyaw[3], v[0]])

        if nmpc.x0 is None: nmpc.ocp.init_x(x0)  # init ocp state matrix at first control loop
        nmpc.set_x0(x0)

        ## reference
        if not system.start_motion:
            if p_hover is None:
                nmpc.set_world_vel_ref([0,0])
                p_des = None
                p_hover = x0[:3]
                if cfg.mission['use_current_z']:
                    nmpc.set_z_ref(x0[2])
                else:
                    nmpc.set_z_ref(zref)
        else:
            if p_des is None:
                p_hover = None
                p_des = x0[:2] + np.array(quat2rot(qyaw) @ np.array([p_des_body[0], p_des_body[1], 0]).reshape(-1,1))[:2].flatten()
                print('current position: ', x0[:2])
                print('setting desired position:', p_des)
            nmpc.set_waypoint(p_des, vref)

        ## params
        nmpc.set_colpred_flag(system.nmpc_flag)

        ## run mpc
        fail = nmpc.solve()
        if fail:
            print('NMPC FAILED. SENDING 0 VEL COMMAND.')
            system.send_commands([0,0,0,0])
            system.start = False
            system.start_motion = False
            system.nmpc_flag = False
            p_hover = None
            p_des = None

        # print('---')
        nmpc_path = []
        for i in range(nmpc.N + 1):
            x = nmpc.ocp.solver.get(i, 'x')
            nmpc_path.append(x[[0,1,2,3,4]])
            # print(nmpc.ocp.solver.get(i, 'x'))
        system.send_path(nmpc_path)

        ## control
        u = nmpc.get_u()
        vx = nmpc.ocp.solver.get(1, 'x')[-1]
        # vx = u[0]*ts + x0[-1]
        cmd = [vx, u[1], u[2]]
        cmd_clipped = np.clip(cmd, -1, 1)  # just in case
        if not fail:
            system.send_commands(cmd_clipped)

        ## log
        t = time.time()
        logx = x0.reshape(1,-1)
        logu = u.reshape(1,-1)
        logy = nmpc.y[2:5,0].reshape(1,-1)
        loginfo = np.array([t, nmpc.ocp.get_t(), nmpc.p[0], nmpc.ocp.solver.get_cost()]).reshape(1,-1)
        logdata = np.hstack([
            loginfo,
            logx,
            logu,
            [nmpc.eval(nmpc.N)],
            logy,
        ])
        with open(logfile, 'a') as flog:
            np.savetxt(flog, logdata, fmt='%3f', delimiter=' ', newline='\n')
        with open(hlogfile, 'a') as flog:
            for i in range(1, nmpc.N+1):
                logdata = np.array([t, i]).reshape(1,-1)
                x = nmpc.ocp.solver.get(i, 'x').reshape(1,-1)
                eval = nmpc.eval(i)
                logdata = np.hstack([logdata, x, [eval]])
                np.savetxt(flog, logdata, fmt='%3f', delimiter=' ', newline='\n')

        ## wait
        time_spent = time.time() + tic
        if time_spent < ts:
            rospy.sleep(ts - time_spent)

    print('ROSPY DEAD. EXITING.')
    system.send_commands([0,0,0,0])
    exit(0)
