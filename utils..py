import numpy as np

'''
convert a unit quaternion to angle/axis representation
'''                                                                                                            
def quat2axang(q): 
    s = np.linalg.norm(q[1:4])
    if s >= 10*np.finfo(float).eps:#10*np.finfo(q.dtype).eps:
        vector = q[1:4]/s
        theta = 2*np.arctan2(s,q[0])
    else:
        vector = np.array([0,0,1])
        theta = 0
    
    axang = np.hstack((vector,theta))
    return axang
    
    
'''
multiply two quaternions (numpy arrays)
'''
def quatmultiply(q1, q2):
    # scalar = s1*s2 - dot(v1,v2)
    scalar = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]

    # vector = s1*v2 + s2*v1 + cross(v1,v2)
    vector = np.array([q1[0]*q2[1], q1[0]*q2[2], q1[0]*q2[3]]) + \
             np.array([q2[0]*q1[1], q2[0]*q1[2], q2[0]*q1[3]]) + \
             np.array([ q1[2]*q2[3]-q1[3]*q2[2], \
                        q1[3]*q2[1]-q1[1]*q2[3], \
                        q1[1]*q2[2]-q1[2]*q2[1]])

    rslt = np.hstack((scalar, vector))
    return rslt
