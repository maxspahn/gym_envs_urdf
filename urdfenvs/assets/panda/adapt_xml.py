import xml.etree.ElementTree as ET

from urdfenvs.scene_examples.obstacles import sphereObst1, wall_obstacles, sphereObst2

 
# Passing the path of the
# xml document to enable the
# parsing process
tree = ET.parse('panda_scene.xml')
worldbody = tree.getroot().find('worldbody')


obstacles = [sphereObst1, sphereObst2, wall_obstacles[0]]

counter = 0
for obst in obstacles:
    counter += 1
    geom_values = {
        'name': obst.name()+str(counter),
        'type': obst.type(),
        'rgba': " ".join([str(i) for i in obst.rgba()]),
        'pos': " ".join([str(i) for i in obst.position()]),
        'size': " ".join([str(i) for i in obst.size()]),
    }
    my_element = ET.SubElement(worldbody, 'geom', geom_values)
tree.write("panda_scene_with_box.xml")

