import pybullet as p

class LinkIdNotFoundException(Exception):
    pass

def link_name_to_link_id(body_id: int, link_name: str):
    for i in range(p.getNumJoints(body_id)):
        joint_info = p.getJointInfo(body_id, i)
        if link_name == joint_info[12].decode('utf-8'):
            return i
    raise LinkIdNotFoundException(link_name)
