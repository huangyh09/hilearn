# some basic functions

import numpy as np

def id_mapping(IDs1, IDs2, uniq_ref_only=True):
    """
    Mapping IDs2 to IDs1. IDs1 (ref id) can have repeat values, but IDs2 need 
    to only contain unique ids.
    Therefore, IDs2[rv_idx] will be the same as IDs1.
    
    Parameters
    ----------
    IDs1 : array_like or list
        ids for reference.
    IDs2 : array_like or list
        ids waiting to map.
        
    Returns
    -------
    RV_idx : array_like, the same length of IDs1
        The index for IDs2 mapped to IDs1. If an id in IDs1 does not exist 
        in IDs2, then return a None for that id.
    """
    idx1 = np.argsort(IDs1)
    idx2 = np.argsort(IDs2)
    RV_idx1, RV_idx2 = [], []
    
    i, j = 0, 0
    while i < len(idx1):
        if j == len(idx2) or IDs1[idx1[i]] < IDs2[idx2[j]]:
            RV_idx1.append(idx1[i])
            RV_idx2.append(None)
            i += 1
        elif IDs1[idx1[i]] == IDs2[idx2[j]]:
            RV_idx1.append(idx1[i])
            RV_idx2.append(idx2[j])
            i += 1
            if uniq_ref_only: j += 1
        elif IDs1[idx1[i]] > IDs2[idx2[j]]:
            j += 1
            
    origin_idx = np.argsort(RV_idx1)
    RV_idx = np.array(RV_idx2)[origin_idx]
    return RV_idx



# def id_mapping(ref_ids, new_ids):
#     """
#     Mapping new_ids to ref_ids. The latter should only contain unique ids.
    
#     Parameters
#     ----------
#     ref_ids : array_like or list
#         ids for reference.
#     new_ids : array_like or list
#         ids waiting to map.
        
#     Returns
#     -------
#     RV_idx : array_like, the same length of new_ids
#         The index for new_ids mapped to ref_ids. If an id in new_ids does 
#         not exist in ref_ids, then return a None for that id.
#     """
#     idx1 = np.argsort(ref_ids)
#     idx2 = np.argsort(new_ids)
#     RV_idx1, RV_idx2 = [], []
    
#     i, j = 0, 0
#     while i < len(idx1):
#         if j == len(idx2) or IDs1[idx1[i]] < IDs2[idx2[j]]:
#             RV_idx1.append(idx1[i])
#             RV_idx2.append(None)
#             i += 1
#         elif IDs1[idx1[i]] == IDs2[idx2[j]]:
#             RV_idx1.append(idx1[i])
#             RV_idx2.append(idx2[j])
#             i += 1
#             j += 1
#         elif IDs1[idx1[i]] > IDs2[idx2[j]]:
#             j += 1
            
#     origin_idx = np.argsort(RV_idx1)
#     RV_idx = np.array(RV_idx2)[origin_idx]
#     return RV_idx