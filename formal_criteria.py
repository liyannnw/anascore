#!/usr/bin/env python
# coding=utf-8
from collections import Counter

def is_formal_analogy(quad):
    """
    Checks whether a given analogy (quad) satisfies formal analogy criteria.
    
    A formal analogy satisfies three conditions:
        1. Length Constraint: The difference in length between A and B should be the same as between C and D.
        2. Token Distribution Postulate: The difference in token occurrences follow the same pattern between two ratios.
        3. Edit Distance Equivalence: The edit distance between A and B should match C and D.

    Args:
        quad (list): A list containing four terms [A, B, C, D] forming an analogy.

    Returns:
        bool: True if all three conditions are met, otherwise False.
    """


    slc=length_constraint(quad)
    wdp = token_distribution_postulate([list(term) for term in quad])
    edse = edit_dist_equiv(quad)

    return sum([slc,wdp,edse]) ==3


def length_constraint(quad):
    A,B,C,D = quad
    # inputs are token lists
    #B-A=D-C
    return len(D)-len(C) == len(B)-len(A)

def lcs_algo(S1, S2, m, n):
    #https://www.programiz.com/dsa/longest-common-subsequence
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # Building the mtrix in bottom-up way
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif S1[i - 1] == S2[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    index = L[m][n]

    lcs_algo = [""] * (index + 1)
    lcs_algo[index] = ""

    i = m
    j = n
    while i > 0 and j > 0:

        if S1[i - 1] == S2[j - 1]:
            lcs_algo[index - 1] = S1[i - 1]
            i -= 1
            j -= 1
            index -= 1

        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # # Printing the sub sequences
    # print("S1 : " + S1 + "\nS2 : " + S2)
    # print("LCS: " + "".join(lcs_algo))

    return lcs_algo

def lcs_sim(x,y):
    return len(lcs_algo(x, y, len(x), len(y)))-1

def lcs_dist(x,y):
    return len(x) + len(y) - 2 * lcs_sim(x,y)


# def AinBC(A,B,C, corr_func=None):
#     if corr_func:
#         corr_idxAB = corr.get(l=A,r=B)
#         corr_idxAC = corr.get(l=A,r=C)
#         # print(A,B)
#         # print(A,C)
#         # print(corr_idxAB)
#         # print(corr_idxAC)
#         if corr_idxAB and corr_idxAC:
#             return set(list(zip(*corr_idxAB))[0])| set(list(zip(*corr_idxAC))[0]) == set(range(len(A)))

#             # print(set(list(zip(*corr_idxAB))[0])| set(list(zip(*corr_idxAC))[0]) == set(range(len(A))))
#         else:
#             return False
#         #     print(False)

#         # return set(list(zip(*corr_idxAB))[0])| set(list(zip(*corr_idxAC))[0]) == set(range(len(A)))
    
#     else:
#         return lcs_sim(A,B) + lcs_sim(A,C) >= len(A) #TODO: it is too weak


# def token_distribution_postulate(quad,corr_func=None):
#     A,B,C,D = quad
#     return  AinBC(A,B,C, corr_func=corr_func) and AinBC(B,A,D, corr_func=corr_func) and AinBC(C,A,D, corr_func=corr_func) and AinBC(D,B,C, corr_func=corr_func)


def tdp(Aocc=None,Bocc=None,Cocc=None,Docc=None):
    is_true=True

    for tok, D_occ in Docc.items():
        A_occ = Aocc[tok] if tok in Aocc else 0
        B_occ = Bocc[tok] if tok in Bocc else 0
        C_occ = Cocc[tok] if tok in Cocc else 0

        D_occ_ = B_occ-A_occ+C_occ
        if D_occ_ != D_occ:
            is_true=False
            break
    
    return is_true


def token_distribution_postulate(quad):
    tok2occs = [dict(Counter(term)) for term in quad]

    return tdp(Aocc=tok2occs[0],Bocc=tok2occs[1],Cocc=tok2occs[2],Docc=tok2occs[3]) and \
        tdp(Aocc=tok2occs[1],Bocc=tok2occs[0],Cocc=tok2occs[3],Docc=tok2occs[2]) and \
            tdp(Aocc=tok2occs[2],Bocc=tok2occs[0],Cocc=tok2occs[3],Docc=tok2occs[1]) and \
                tdp(Aocc=tok2occs[3],Bocc=tok2occs[1],Cocc=tok2occs[2],Docc=tok2occs[0])

    


def edit_dist_equiv(quad):
    A,B,C,D = quad

    dAB = lcs_dist(A, B)
    dCD = lcs_dist(C, D)

    dAC = lcs_dist(A, C)
    dBD = lcs_dist(B, D)

    return dAB==dCD  and dAC == dBD



