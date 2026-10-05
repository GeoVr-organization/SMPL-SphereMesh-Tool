"""Automated authoring of sphere-mesh topologies for SMPL bodies.

``bumpers``   the collision primitives a connection list generates, their exact signed
              distance, and the adjacency graph the cloth solver's watchdog walks
``anatomy``   measurements of an SMPL body a placement rule can be expressed in
``validate``  the checks a candidate topology has to pass, including a pose sweep
``export``    writing a topology out as tool JSON and engine text
"""
