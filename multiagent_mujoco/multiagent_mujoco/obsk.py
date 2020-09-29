import itertools
import numpy as np


class Node:
    def __init__(
        self,
        label,
        qpos_ids,
        qvel_ids,
        act_ids,
        body_fn=None,
        bodies=None,
        extra_obs=None,
    ):
        self.label = label
        self.qpos_ids = qpos_ids
        self.qvel_ids = qvel_ids
        self.act_ids = act_ids
        self.bodies = bodies
        self.extra_obs = {} if extra_obs is None else extra_obs
        self.body_fn = body_fn
        pass

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label


class HyperEdge:
    def __init__(self, *edges):
        self.edges = set(edges)

    def __contains__(self, item):
        return item in self.edges

    def __str__(self):
        return "HyperEdge({})".format(self.edges)

    def __repr__(self):
        return "HyperEdge({})".format(self.edges)


def get_joints_at_kdist(
    agent_id, agent_partitions, hyperedges, k=0, kagents=False,
):
    """ Identify all joints at distance <= k from agent agent_id

    :param agent_id: id of agent to be considered
    :param agent_partitions: list of joint tuples in order of agentids
    :param edges: list of tuples (joint1, joint2)
    :param k: kth degree
    :param kagents: True (observe all joints of an agent if a single one is) or False (individual joint granularity)
    :return:
        dict with k as key, and list of joints at that distance
    """
    assert not kagents, "kagents not implemented!"

    agent_joints = agent_partitions[agent_id]

    def _adjacent(lst, kagents=False):
        # return all sets adjacent to any element in lst
        ret = set([])
        for l in lst:
            ret = ret.union(
                set(
                    itertools.chain(
                        *[e.edges.difference({l}) for e in hyperedges if l in e]
                    )
                )
            )
        return ret

    seen = set([])
    new = set([])
    k_dict = {}
    for _k in range(k + 1):
        if not _k:
            new = set(agent_joints)
        else:
            print(hyperedges)
            new = _adjacent(new) - seen
        seen = seen.union(new)
        k_dict[_k] = sorted(list(new), key=lambda x: x.label)
    return k_dict


def build_obs(env, k_dict, k_categories, global_dict, global_categories, vec_len=None):
    """Given a k_dict from get_joints_at_kdist, extract observation vector.

    :param k_dict: k_dict
    :param qpos: qpos numpy array
    :param qvel: qvel numpy array
    :param vec_len: if None no padding, else zero-pad to vec_len
    :return:
    observation vector
    """

    # TODO: This needs to be fixed, it was designed for half-cheetah only!
    # if add_global_pos:
    #    obs_qpos_lst.append(global_qpos)
    #    obs_qvel_lst.append(global_qvel)

    body_set_dict = {}
    obs_lst = []
    # Add parts attributes
    for k in sorted(list(k_dict.keys())):
        cats = k_categories[k]
        for _t in k_dict[k]:
            for c in cats:
                if c in _t.extra_obs:
                    items = _t.extra_obs[c](env).tolist()
                    obs_lst.extend(items if isinstance(items, list) else [items])
                else:
                    if c in [
                        "qvel",
                        "qpos",
                    ]:  # this is a "joint position/velocity" item
                        items = getattr(env.sim.data, c)[
                            getattr(_t, "{}_ids".format(c))
                        ]
                        obs_lst.extend(items if isinstance(items, list) else [items])
                    elif c in ["qfrc_actuator"]:  # this is a "vel position" item
                        items = getattr(env.sim.data, c)[
                            getattr(_t, "{}_ids".format("qvel"))
                        ]
                        obs_lst.extend(items if isinstance(items, list) else [items])
                    elif c in [
                        "cvel",
                        "cinert",
                        "cfrc_ext",
                    ]:  # this is a "body position" item
                        if _t.bodies is not None:
                            for b in _t.bodies:
                                if c not in body_set_dict:
                                    body_set_dict[c] = set()
                                if b not in body_set_dict[c]:
                                    items = getattr(env.sim.data, c)[b].tolist()
                                    items = getattr(_t, "body_fn", lambda _id, x: x)(
                                        b, items
                                    )
                                    obs_lst.extend(
                                        items if isinstance(items, list) else [items]
                                    )
                                    body_set_dict[c].add(b)
    # Add global attributes
    body_set_dict = {}
    for c in global_categories:
        if c in ["qvel", "qpos"]:  # this is a "joint position" item
            for j in global_dict.get("joints", []):
                items = getattr(env.sim.data, c)[getattr(j, "{}_ids".format(c))]
                obs_lst.extend(ids if isinstance(items, list) else [items])
        else:
            for b in global_dict.get("bodies", []):
                if c not in body_set_dict:
                    body_set_dict[c] = set()
                if b not in body_set_dict[c]:
                    obs_lst.extend(getattr(env.sim.data, c)[b].tolist())
                    body_set_dict[c].add(b)

    if vec_len is not None:
        pad = np.array((vec_len - len(obs_lst)) * [0])
        if len(pad):
            return np.concatenate([np.array(obs_lst), pad])
    return np.array(obs_lst)


def build_actions(agent_partitions, k_dict):
    # Composes agent actions output from networks
    # into coherent joint action vector to be sent to the env.
    pass


def get_parts_and_edges(label, partitioning):
    if label in ["half_cheetah", "HalfCheetah-v2"]:

        # define Mujoco graph
        bthigh = Node("bthigh", -6, -6, 0)
        bshin = Node("bshin", -5, -5, 1)
        bfoot = Node("bfoot", -4, -4, 2)
        fthigh = Node("fthigh", -3, -3, 3)
        fshin = Node("fshin", -2, -2, 4)
        ffoot = Node("ffoot", -1, -1, 5)

        edges = [
            HyperEdge(bfoot, bshin),
            HyperEdge(bshin, bthigh),
            HyperEdge(bthigh, fthigh),
            HyperEdge(fthigh, fshin),
            HyperEdge(fshin, ffoot),
        ]

        root_x = Node("root_x", 0, 0, -1, extra_obs={"qpos": lambda env: np.array([])})
        root_z = Node("root_z", 1, 1, -1)
        root_y = Node("root_y", 2, 2, -1)
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "2x3":
            parts = [(bfoot, bshin, bthigh), (ffoot, fshin, fthigh)]
        elif partitioning == "6x1":
            parts = [(bfoot,), (bshin,), (bthigh,), (ffoot,), (fshin,), (fthigh,)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Ant-v2"]:

        # define Mujoco graph
        torso = 1
        front_left_leg = 2
        aux_1 = 3
        ankle_1 = 4
        front_right_leg = 5
        aux_2 = 6
        ankle_2 = 7
        back_leg = 8
        aux_3 = 9
        ankle_3 = 10
        right_back_leg = 11
        aux_4 = 12
        ankle_4 = 13

        hip1 = Node(
            "hip1",
            -8,
            -8,
            2,
            bodies=[torso, front_left_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  #
        #            ,extra_obs={"cfrc_ext": lambda env: np.clip(np.concatenate([env.sim.data.cfrc_ext[torso],
        #                                                                       env.sim.data.cfrc_ext[front_left_leg]]), -1, 1)})
        ankle1 = Node(
            "ankle1",
            -7,
            -7,
            3,
            bodies=[front_left_leg, aux_1, ankle_1],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        # extra_obs={"cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[-7], -1, 1)})
        hip2 = Node(
            "hip2",
            -6,
            -6,
            4,
            bodies=[torso, front_right_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        # extra_obs={"cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[-6], -1, 1)})
        ankle2 = Node(
            "ankle2",
            -5,
            -5,
            5,
            bodies=[front_right_leg, aux_2, ankle_2],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        # extra_obs={"cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[-5], -1, 1)})
        hip3 = Node(
            "hip3",
            -4,
            -4,
            6,
            bodies=[torso, back_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        # extra_obs={"cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[-4], -1, 1)})
        ankle3 = Node(
            "ankle3",
            -3,
            -3,
            7,
            bodies=[back_leg, aux_3, ankle_3],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        # extra_obs={"cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[-3], -1, 1)})
        hip4 = Node(
            "hip4",
            -2,
            -2,
            0,
            bodies=[torso, right_back_leg],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        # extra_obs={"cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[-2], -1, 1)})
        ankle4 = Node(
            "ankle4",
            -1,
            -1,
            1,
            bodies=[right_back_leg, aux_4, ankle_4],
            body_fn=lambda _id, x: np.clip(x, -1, 1).tolist(),
        )  # ,
        # extra_obs={"cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[-1], -1, 1)})

        edges = [
            HyperEdge(ankle4, hip4),
            HyperEdge(ankle1, hip1),
            HyperEdge(ankle2, hip2),
            HyperEdge(ankle3, hip3),
            HyperEdge(hip4, hip1, hip2, hip3),
        ]

        free_joint = Node(
            "free",
            0,
            0,
            -1,
            extra_obs={
                "qpos": lambda env: env.sim.data.qpos[:7],
                "qvel": lambda env: env.sim.data.qvel[:6],
                "cfrc_ext": lambda env: np.clip(env.sim.data.cfrc_ext[0:1], -1, 1),
            },
        )
        globals = {"joints": [free_joint]}

        if partitioning == "2x4":  # neighbouring legs together
            parts = [(hip1, ankle1, hip2, ankle2), (hip3, ankle3, hip4, ankle4)]
        elif partitioning == "2x4d":  # diagonal legs together
            parts = [(hip1, ankle1, hip3, ankle3), (hip2, ankle2, hip4, ankle4)]
        elif partitioning == "4x2":
            parts = [(hip1, ankle1), (hip2, ankle2), (hip3, ankle3), (hip4, ankle4)]
        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Hopper-v2"]:

        # define Mujoco-Graph
        thigh_joint = Node(
            "thigh_joint",
            -3,
            -3,
            0,
            extra_obs={
                "qvel": lambda env: np.clip(np.array([env.sim.data.qvel[-3]]), -10, 10)
            },
        )
        leg_joint = Node(
            "leg_joint",
            -2,
            -2,
            1,
            extra_obs={
                "qvel": lambda env: np.clip(np.array([env.sim.data.qvel[-2]]), -10, 10)
            },
        )
        foot_joint = Node(
            "foot_joint",
            -1,
            -1,
            2,
            extra_obs={
                "qvel": lambda env: np.clip(np.array([env.sim.data.qvel[-1]]), -10, 10)
            },
        )

        edges = [HyperEdge(foot_joint, leg_joint), HyperEdge(leg_joint, thigh_joint)]

        root_x = Node(
            "root_x",
            0,
            0,
            -1,
            extra_obs={
                "qpos": lambda env: np.array([]),
                "qvel": lambda env: np.clip(np.array([env.sim.data.qvel[1]]), -10, 10),
            },
        )
        root_z = Node(
            "root_z",
            1,
            1,
            -1,
            extra_obs={
                "qvel": lambda env: np.clip(np.array([env.sim.data.qvel[1]]), -10, 10)
            },
        )
        root_y = Node(
            "root_y",
            2,
            2,
            -1,
            extra_obs={
                "qvel": lambda env: np.clip(np.array([env.sim.data.qvel[2]]), -10, 10)
            },
        )
        globals = {"joints": [root_x, root_y, root_z]}

        if partitioning == "3x1":
            parts = [(thigh_joint,), (leg_joint,), (foot_joint,)]

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Humanoid-v2", "HumanoidStandup-v2"]:

        # define Mujoco-Graph
        abdomen_y = Node(
            "abdomen_y", -16, -16, 0
        )  # act ordering bug in env -- double check!
        abdomen_z = Node("abdomen_z", -17, -17, 1)
        abdomen_x = Node("abdomen_x", -15, -15, 2)
        right_hip_x = Node("right_hip_x", -14, -14, 3)
        right_hip_z = Node("right_hip_z", -13, -13, 4)
        right_hip_y = Node("right_hip_y", -12, -12, 5)
        right_knee = Node("right_knee", -11, -11, 6)
        left_hip_x = Node("left_hip_x", -10, -10, 7)
        left_hip_z = Node("left_hip_z", -9, -9, 8)
        left_hip_y = Node("left_hip_y", -8, -8, 9)
        left_knee = Node("left_knee", -7, -7, 10)
        right_shoulder1 = Node("right_shoulder1", -6, -6, 11)
        right_shoulder2 = Node("right_shoulder2", -5, -5, 12)
        right_elbow = Node("right_elbow", -4, -4, 13)
        left_shoulder1 = Node("left_shoulder1", -3, -3, 14)
        left_shoulder2 = Node("left_shoulder2", -2, -2, 15)
        left_elbow = Node("left_elbow", -1, -1, 16)

        edges = [
            HyperEdge(abdomen_x, abdomen_y, abdomen_z),
            HyperEdge(right_hip_x, right_hip_y, right_hip_z),
            HyperEdge(left_hip_x, left_hip_y, left_hip_z),
            HyperEdge(left_elbow, left_shoulder1, left_shoulder2),
            HyperEdge(right_elbow, right_shoulder1, right_shoulder2),
            HyperEdge(left_knee, left_hip_x, left_hip_y, left_hip_z),
            HyperEdge(right_knee, right_hip_x, right_hip_y, right_hip_z),
            HyperEdge(left_shoulder1, left_shoulder2, abdomen_x, abdomen_y, abdomen_z),
            HyperEdge(
                right_shoulder1, right_shoulder2, abdomen_x, abdomen_y, abdomen_z
            ),
            HyperEdge(
                abdomen_x, abdomen_y, abdomen_z, left_hip_x, left_hip_y, left_hip_z
            ),
            HyperEdge(
                abdomen_x, abdomen_y, abdomen_z, right_hip_x, right_hip_y, right_hip_z
            ),
        ]

        if (
            partitioning == "2x8"
        ):  # 17 in total, so one action is a dummy (to be handled by pymarl)
            # isolate upper and lower body
            parts = [
                (
                    left_shoulder1,
                    left_shoulder2,
                    abdomen_x,
                    abdomen_y,
                    abdomen_z,
                    right_shoulder1,
                    right_shoulder2,
                    right_elbow,
                    left_elbow,
                ),
                (
                    left_hip_x,
                    left_hip_y,
                    left_hip_z,
                    right_hip_x,
                    right_hip_y,
                    right_hip_z,
                    right_knee,
                    left_knee,
                ),
            ]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges

    elif label in ["Reacher-v2"]:

        # define Mujoco-Graph
        body0 = 1
        body1 = 2
        fingertip = 3
        joint0 = Node(
            "joint0",
            -4,
            -4,
            0,
            bodies=[body0, body1],
            extra_obs={
                "qpos": (
                    lambda env: np.array(
                        [np.sin(env.sim.data.qpos[-4]), np.cos(env.sim.data.qpos[-4])]
                    )
                )
            },
        )
        joint1 = Node(
            "joint1",
            -3,
            -3,
            1,
            bodies=[body1, fingertip],
            extra_obs={
                "fingertip_dist": (
                    lambda env: env.get_body_com("fingertip")
                    - env.get_body_com("target")
                ),
                "qpos": (
                    lambda env: np.array(
                        [np.sin(env.sim.data.qpos[-3]), np.cos(env.sim.data.qpos[-3])]
                    )
                ),
            },
        )
        edges = [HyperEdge(joint0, joint1)]

        worldbody = 0
        target = 4
        target_x = Node(
            "target_x", -2, -2, -1, extra_obs={"qvel": (lambda env: np.array([]))}
        )
        target_y = Node(
            "target_y", -1, -1, -1, extra_obs={"qvel": (lambda env: np.array([]))}
        )
        globals = {"bodies": [worldbody, target], "joints": [target_x, target_y]}

        if partitioning == "2x1":
            # isolate upper and lower arms
            parts = [(joint0,), (joint1,)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges, globals

    elif label in ["Swimmer-v2"]:

        # define Mujoco-Graph
        joint0 = Node("rot2", -2, -2, 0)  # TODO: double-check ids
        joint1 = Node("rot3", -1, -1, 1)

        edges = [HyperEdge(joint0, joint1)]

        if partitioning == "2x1":
            # isolate upper and lower body
            parts = [(joint0,), (joint1,)]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges

    elif label in ["Walker2d-v2"]:

        # define Mujoco-Graph
        thigh_joint = Node("thigh_joint", -6, -6, 0)
        leg_joint = Node("leg_joint", -5, -5, 1)
        foot_joint = Node("foot_joint", -4, -4, 2)
        thigh_left_joint = Node("thigh_left_joint", -3, -3, 3)
        leg_left_joint = Node("leg_left_joint", -2, -2, 4)
        foot_left_joint = Node("foot_left_joint", -1, -1, 5)

        edges = [
            HyperEdge(foot_joint, leg_joint),
            HyperEdge(leg_joint, thigh_joint),
            HyperEdge(foot_left_joint, leg_left_joint),
            HyperEdge(leg_left_joint, thigh_left_joint),
            HyperEdge(thigh_joint, thigh_left_joint),
        ]

        if partitioning == "2x3":
            # isolate upper and lower body
            parts = [
                (foot_joint, leg_joint, thigh_joint),
                (foot_left_joint, leg_left_joint, thigh_left_joint,),
            ]
            # TODO: There could be tons of decompositions here

        else:
            raise Exception("UNKNOWN partitioning config: {}".format(partitioning))

        return parts, edges
