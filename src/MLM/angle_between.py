
import numpy as np 


def angle_between(v1 : np.array,
                  v2 : np.array) -> float:
    """returns angle between two vectors in degrees

    Args:
        v1 (np.array): vector_1
        v2 (np.array): vector_2

    Returns:
        float: angle between vectors in degreee
    """
    return np.rad2deg(np.arccos((np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))))
