from typing import List

from .sim3 import Sim3


def refine_sim3_loop(sim3_loop: List[Sim3]) -> List[Sim3]:
    sim3_seq = sim3_loop[:-1]
    constraint = sim3_loop[-1].inv()
    sim3_seq = refine_sim3_sequence(sim3_seq, constraint)
    sim3_loop[:-1] = sim3_seq
    return sim3_loop


def refine_sim3_sequence(sim3_seq: List[Sim3], constraint: Sim3) -> List[Sim3]:
    return sim3_seq


def refine_sim3_loop_with_interpolation(sim3_loop: List[Sim3]) -> List[Sim3]:
    #Here we expect: S_{0,1}S_{1,2},...,S_{n-1,n}S_{n,0} = I
    #Hence, S_{0,1}S_{1,2},...,S_{n-1,n} = S_{n,0}^-1
    #If the chunks were already aligned. S{i,i+1} \approx I from i in [0, n-1]
    #Hence, I = S_{n,0}^-1
    #When optimizing, we expect S_{0,1}'S_{1,2}',...,S_{n-1,n}'S_{n,0} \approx I and S{i,i+1}' \approx I
    # We take the following assumption S{i,i+1}' = S'
    # Thus S'^n S{n,0} \approx I
    # Finally, S' = S{n,0}^{1/n}
    sim3_seq = sim3_loop[:-1]
    constraint = sim3_loop[-1]
    n = len(sim3_seq)
    print("refine_sim3_loop_with_interpolation", n)
    s_est = constraint.nthroot(n)
    sim3_loop[:-1] = [s_est.copy() for _ in range(n)]
    return sim3_loop
