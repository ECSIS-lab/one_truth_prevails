from joblib import Parallel, delayed
import sys
import gc
import random
from tqdm import tqdm
import pickle
import time
import copy
from collections import deque
import heapq
import math
import csv
import pandas as pd

candidates =[
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
]

dp_candidates_list =[
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
]

dq_candidates_list =[
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
]

target_key_length = 128 * 4

eval_count = 0
start_time = 0
return_time = 0

eval_time = 0

def tau(x):
    i = 0
    while x & 1 == 0:
        x >> 1
        i += 1
    return i


def next_slice_check(slice, p_so_far, q_so_far, dp_so_far, dq_so_far, next_index, kp, kq, tkp, tkq, e, N, mode):
    p_candidate = slice[0]
    q_candidate = slice[1]
    dp_candidate = slice[2]
    dq_candidate = slice[3]
    c1 = ((N - p_so_far * q_so_far) >> next_index-1) & 1
    c2 = ((kp * (p_so_far - 1) + 1 - e * dp_so_far) >> (next_index-1+tkp)) & 1
    c3 = ((kq * (q_so_far - 1) + 1 - e * dq_so_far) >> (next_index-1+tkq)) & 1

    if mode == 0:
        if p_candidate ^ q_candidate == c1 and dp_candidate ^ p_candidate == c2 and dq_candidate ^ q_candidate == c3:
            return True
        else:
            return False
    elif mode == 1:
        if p_candidate ^ q_candidate == c1 and dp_candidate ^ p_candidate == c2:
            return True
        else:
            return False
    elif mode == 2:
        if p_candidate ^ q_candidate == c1 and dq_candidate ^ q_candidate == c3:
            return True
        else:
            return False
    elif mode == 3:
        if p_candidate ^ q_candidate == c1:
            return True
        else:
            return False

def transform_to_bit_list(str_key):
    bit_list_key = []
    for i in range(len(str_key)):
        partial_4bit = str_key[i]
        partial_4bit = format(int(partial_4bit, 16), "04b")
        for j in range(4):
            bit_list_key.append(int(partial_4bit[j]))
    return bit_list_key

def parameter_so_far(current_slice_4bit, current_index, parameter_so_far_list, tkp, tkq):
    p_so_far = parameter_so_far_list[0]
    q_so_far = parameter_so_far_list[1]
    dp_so_far = parameter_so_far_list[2]
    dq_so_far = parameter_so_far_list[3]

    p_so_far += ((current_slice_4bit[0][3] << (current_index-4)) + (current_slice_4bit[0][2] << (current_index-3)) + (current_slice_4bit[0][1] << (current_index-2)) + (current_slice_4bit[0][0] << (current_index-1)))
    q_so_far += ((current_slice_4bit[1][3] << (current_index-4)) + (current_slice_4bit[1][2] << (current_index-3)) + (current_slice_4bit[1][1] << (current_index-2)) + (current_slice_4bit[1][0] << (current_index-1)))
    dp_so_far += ((current_slice_4bit[2][3] << (current_index+tkp-4)) + (current_slice_4bit[2][2] << (current_index+tkp-3)) + (current_slice_4bit[2][1] << (current_index+tkp-2)) + (current_slice_4bit[2][0] << (current_index+tkp-1)))
    dq_so_far += ((current_slice_4bit[3][3] << (current_index+tkq-4)) + (current_slice_4bit[3][2] << (current_index+tkq-3)) + (current_slice_4bit[3][1] << (current_index+tkq-2)) + (current_slice_4bit[3][0] << (current_index+tkq-1)))

    return p_so_far, q_so_far, dp_so_far, dq_so_far

def result_is_ok(parameter_so_far_list, kp, kq):
    (p, q, dp, dq) = (parameter_so_far_list[0], parameter_so_far_list[1], parameter_so_far_list[2], parameter_so_far_list[3])
    if p * q == N and e*dp == 1 + kp * (p-1) and e*dq == 1 + kq * (q-1):
        return True
    else:
        return False

def next_slice_check_4bit_wise(p_candidate_4bit, q_candidate_4bit, dp_candidate_4bit, dq_candidate_4bit, p_so_far, q_so_far, dp_so_far, dq_so_far, next_index, kp, kq, tkp, tkq, e, N, is_overflow, c1, c2, c3):
    mode = 0
    if is_overflow:
        mode = 3 if tkp >= 4 and tkq >= 4 else 2 if tkp >= 4 else 1 if tkq >= 4 else 0

    eval_1 = (p_candidate_4bit[3] ^ q_candidate_4bit[3] == c1)
    eval_2 = (dp_candidate_4bit[3] ^ p_candidate_4bit[3] == c2) or (mode == 2) or (mode == 3)
    eval_3 = (dq_candidate_4bit[3] ^ q_candidate_4bit[3] == c3) or (mode == 1) or (mode == 3)
    if (eval_1 and eval_2 and eval_3):
        p_so_far += p_candidate_4bit[3] << (next_index-1)
        q_so_far += q_candidate_4bit[3] << (next_index-1)
        dp_so_far += dp_candidate_4bit[3] << (next_index+tkp-1)
        dq_so_far += dq_candidate_4bit[3] << (next_index+tkq-1)
        next_index += 1
        c1 = ((N - p_so_far * q_so_far) >> next_index-1) & 1
        c2 = ((kp * (p_so_far - 1) + 1 - e * dp_so_far) >> (next_index-1+tkp)) & 1
        c3 = ((kq * (q_so_far - 1) + 1 - e * dq_so_far) >> (next_index-1+tkq)) & 1
    else:
        return False
    
    if is_overflow:
        mode = 3 if tkp >= 3 and tkq >= 3 else 2 if tkp >= 3 else 1 if tkq >= 3 else 0

    eval_1 = (p_candidate_4bit[2] ^ q_candidate_4bit[2] == c1)
    eval_2 = (dp_candidate_4bit[2] ^ p_candidate_4bit[2] == c2) or (mode == 2) or (mode == 3)
    eval_3 = (dq_candidate_4bit[2] ^ q_candidate_4bit[2] == c3) or (mode == 1) or (mode == 3)
    if (eval_1 and eval_2 and eval_3):
        p_so_far += p_candidate_4bit[2] << (next_index-1)
        q_so_far += q_candidate_4bit[2] << (next_index-1)
        dp_so_far += dp_candidate_4bit[2] << (next_index+tkp-1)
        dq_so_far += dq_candidate_4bit[2] << (next_index+tkq-1)
        next_index += 1
        c1 = ((N - p_so_far * q_so_far) >> next_index-1) & 1
        c2 = ((kp * (p_so_far - 1) + 1 - e * dp_so_far) >> (next_index-1+tkp)) & 1
        c3 = ((kq * (q_so_far - 1) + 1 - e * dq_so_far) >> (next_index-1+tkq)) & 1
    else:
        return False

    if is_overflow:
        mode = 3 if tkp >= 2 and tkq >= 2 else 2 if tkp >= 2 else 1 if tkq >= 2 else 0

    eval_1 = (p_candidate_4bit[1] ^ q_candidate_4bit[1] == c1)
    eval_2 = (dp_candidate_4bit[1] ^ p_candidate_4bit[1] == c2) or (mode == 2) or (mode == 3)
    eval_3 = (dq_candidate_4bit[1] ^ q_candidate_4bit[1] == c3) or (mode == 1) or (mode == 3)
    if (eval_1 and eval_2 and eval_3):
        p_so_far += p_candidate_4bit[1] << (next_index-1)
        q_so_far += q_candidate_4bit[1] << (next_index-1)
        dp_so_far += dp_candidate_4bit[1] << (next_index+tkp-1)
        dq_so_far += dq_candidate_4bit[1] << (next_index+tkq-1)
        next_index += 1
        c1 = ((N - p_so_far * q_so_far) >> next_index-1) & 1
        c2 = ((kp * (p_so_far - 1) + 1 - e * dp_so_far) >> (next_index-1+tkp)) & 1
        c3 = ((kq * (q_so_far - 1) + 1 - e * dq_so_far) >> (next_index-1+tkq)) & 1
    else:
        return False

    if is_overflow:
        mode = 3 if tkp >= 1 and tkq >= 1 else 2 if tkp >= 1 else 1 if tkq >= 1 else 0

    eval_1 = (p_candidate_4bit[0] ^ q_candidate_4bit[0] == c1)
    eval_2 = (dp_candidate_4bit[0] ^ p_candidate_4bit[0] == c2) or (mode == 2) or (mode == 3)
    eval_3 = (dq_candidate_4bit[0] ^ q_candidate_4bit[0] == c3) or (mode == 1) or (mode == 3)
    if (eval_1 and eval_2 and eval_3):
        return True
    else:
        return False

def slice0_4bit_check(p_candidate_4bit, q_candidate_4bit, dp_candidate_4bit, dq_candidate_4bit, dp_initial_value, dq_initial_value, kp, kq, tkp, tkq):
    dp_so_far = dp_initial_value
    dq_so_far = dq_initial_value
    slice0 = [1, 1, 0 if tkp>0 else 1, 0 if tkq>0 else 1] #slice0
    if dp_candidate_4bit[3] != slice0[2] or dq_candidate_4bit[3] != slice0[3]:
        return False
    p_so_far = 1
    q_so_far = 1
    dp_so_far += slice0[2] << tkp
    dq_so_far += slice0[3] << tkq

    slice1 = [p_candidate_4bit[-2], q_candidate_4bit[-2], dp_candidate_4bit[-2], dq_candidate_4bit[-2]]
    if next_slice_check(slice1, p_so_far, q_so_far, dp_so_far, dq_so_far, 2, kp, kq, tkp, tkq, e, N, 0):
        p_so_far += slice1[0] << 1
        q_so_far += slice1[1] << 1
        dp_so_far += slice1[2] << (tkp+1)
        dq_so_far += slice1[3] << (tkq+1)
    else:
        return False
    
    slice2 = [p_candidate_4bit[-3], q_candidate_4bit[-3], dp_candidate_4bit[-3], dq_candidate_4bit[-3]]
    if next_slice_check(slice2, p_so_far, q_so_far, dp_so_far, dq_so_far, 3, kp, kq, tkp, tkq, e, N, 0):
        p_so_far += slice2[0] << 2
        q_so_far += slice2[1] << 2
        dp_so_far += slice2[2] << (tkp+2)
        dq_so_far += slice2[3] << (tkq+2)
    else:
        return False
    
    slice3 = [p_candidate_4bit[-4], q_candidate_4bit[-4], dp_candidate_4bit[-4], dq_candidate_4bit[-4]]
    if next_slice_check(slice3, p_so_far, q_so_far, dp_so_far, dq_so_far, 4, kp, kq, tkp, tkq, e, N, 0):
        p_so_far += slice3[0] << 3
        q_so_far += slice3[1] << 3
        dp_so_far += slice3[2] << (tkp+3)
        dq_so_far += slice3[3] << (tkq+3)
        return True
    else:
        return False

def errored_d(d, gave_error_index, error_rate):
    d_length = len(d)
    error_index = random.sample(range(d_length), int(d_length*error_rate))
    error_index.sort()
    if gave_error_index!=None:
        error_index = gave_error_index

    pe = map(lambda x: 512 - x * 4, error_index)
    print(list(pe))
    d = list(d)
    for i in error_index:
        d[i] = format(random.randint(0, 15), "x")
    d = "".join(d)
    return d

def generate_initial_node_for_kpair(k_pair, dp_bin_0pad, dq_bin_0pad, candidates_list_in):
    global eval_count
    node_list = []
    kp = k_pair[0]
    kq = k_pair[1]
    tkp = tau(kp)
    tkq = tau(kq)
    slice0_4bit = []
    
    dp_initial_value = 1 if tkp > 0 else 0
    dq_initial_value = 1 if tkq > 0 else 0
    dp_candidates_list_in = copy.deepcopy(candidates_list_in)
    dq_candidates_list_in = copy.deepcopy(candidates_list_in)
    candidates_list_in = [candidate for candidate in candidates_list_in if candidate[-1] == 1]
    
    
    slice0_dp, slice0_dq = 1, 1

    break_loop = False

    initial_dp = dp_bin_0pad[len(dp_bin_0pad)-(tkp+4):len(dp_bin_0pad)-tkp]
    initial_dq = dq_bin_0pad[len(dq_bin_0pad)-(tkq+4):len(dq_bin_0pad)-tkq]

    initial_dp_index = dp_candidates_list_in.index(initial_dp)
    dp_candidates_list_in[0], dp_candidates_list_in[initial_dp_index] = dp_candidates_list_in[initial_dp_index], dp_candidates_list_in[0]

    initial_dq_index = dq_candidates_list_in.index(initial_dq)
    dq_candidates_list_in[0], dq_candidates_list_in[initial_dq_index] = dq_candidates_list_in[initial_dq_index], dq_candidates_list_in[0]
    skip_loop = 0
    for dp_candidate in dp_candidates_list_in:
        for dq_candidate in dq_candidates_list_in:
            for p_candidate in candidates_list_in:
                for q_candidate in candidates_list_in:
                    eval_count += 1

                    if slice0_4bit_check(p_candidate, q_candidate, dp_candidate, dq_candidate, dp_initial_value, dq_initial_value, kp, kq, tkp, tkq):
                        neq = (dp_candidate != initial_dp) + (dq_candidate != initial_dq)
                        if neq == 0:
                            slice0_4bit.append([p_candidate, q_candidate, dp_candidate, dq_candidate, 0, 4])
                            break_loop = True
                            break
                        
                        else:
                            slice0_4bit.append([p_candidate, q_candidate, dp_candidate, dq_candidate, neq, 4])
                            if len(slice0_4bit) + skip_loop == 16:
                                break_loop = True
                                break
                if break_loop:
                    break
            if break_loop:
                break
        if break_loop:
            break

    for s in slice0_4bit:
        s.append([*parameter_so_far(s[0:4], 4, [0,0,(1 if tkp > 0 else 0),(1 if tkq > 0 else 0)], tkp, tkq)])
        s.append(kp)
        s.append(kq)
        s.append([]) #initial num_ignore per slice4bit chain

    return slice0_4bit

def eval_next_node(current_node, dp_bin_0pad, dq_bin_0pad, dp_candidates_list_in, dq_candidates_list_in, score):
    global eval_count

     #[p,q,dp,dq,ignore,bit,[parameter_so_far],kp,kq,ignore_chain]
    ignore_chain = current_node.pop() #[p,q,dp,dq,ignore,bit,[parameter_so_far],kp,kq]
    kq = current_node.pop() #[p,q,dp,dq,ignore,bit,[parameter_so_far],kp]
    kp = current_node.pop() #[p,q,dp,dq,ignore,bit,[parameter_so_far]]
    parameter_so_far_list = current_node.pop() #[p,q,dp,dq,ignore,bit]
    current_bit = current_node.pop() #[p,q,dp,dq,ignore]
    current_ignore = current_node.pop() #[p,q,dp,dq]
    next_bit = current_bit + 1
    tkp = tau(kp)
    tkq = tau(kq)
    ignore_chain.append(current_ignore)
    ignoring_estimation_count = sum(ignore_chain)
    if current_bit == target_key_length:
        if result_is_ok(parameter_so_far_list, kp, kq):

            (p_result, q_result, dp_result, dq_result) = (parameter_so_far_list[0], parameter_so_far_list[1], parameter_so_far_list[2], parameter_so_far_list[3])
            sys.exit()
        else:
            return []
    next_dp = dp_bin_0pad[len(dp_bin_0pad)-(next_bit+tkp+3):len(dp_bin_0pad)-(tkp+next_bit-1)]
    next_dq = dq_bin_0pad[len(dq_bin_0pad)-(next_bit+tkq+3):len(dq_bin_0pad)-(tkq+next_bit-1)]
    next_dp_index = dp_candidates_list_in.index(next_dp)
    next_dq_index = dq_candidates_list_in.index(next_dq)
    dp_candidates_list_in[0], dp_candidates_list_in[next_dp_index] = dp_candidates_list_in[next_dp_index], dp_candidates_list_in[0]
    dq_candidates_list_in[0], dq_candidates_list_in[next_dq_index] = dq_candidates_list_in[next_dq_index], dq_candidates_list_in[0]
    next_slice_candidates=[]
    break_loop = False

    (p_so_far, q_so_far, dp_so_far, dq_so_far) = (parameter_so_far_list[0], parameter_so_far_list[1], parameter_so_far_list[2], parameter_so_far_list[3])
    c1 = ((N - p_so_far * q_so_far) >> next_bit-1) & 1
    c2 = ((kp * (p_so_far - 1) + 1 - e * dp_so_far) >> (next_bit-1+tkp)) & 1
    c3 = ((kq * (q_so_far - 1) + 1 - e * dq_so_far) >> (next_bit-1+tkq)) & 1

    is_overflow = False if (next_bit+tkp+4 <= target_key_length and next_bit+tkq+4 <= target_key_length) else True

    for dp_candidate in dp_candidates_list_in:
        for dq_candidate in dq_candidates_list_in:
            for p_candidate in candidates:
                for q_candidate in candidates:
                    eval_count += 1
                    
                    candidates_is_True = next_slice_check_4bit_wise(p_candidate, 
                                                                    q_candidate, 
                                                                    dp_candidate, 
                                                                    dq_candidate, 
                                                                    p_so_far,
                                                                    q_so_far,
                                                                    dp_so_far,
                                                                    dq_so_far,
                                                                    next_bit, kp, kq, tkp, tkq, e, N, is_overflow, c1, c2, c3)
                    if candidates_is_True:
                        if next_dp == dp_candidate and next_dq == dq_candidate:
                            next_slice_candidates.append([p_candidate, q_candidate, dp_candidate, dq_candidate, 0])
                            if not is_overflow and len(ignore_chain) >= 2 and ignore_chain[-1] + ignore_chain[-2] == 0:
                                match_consecutive = True
                            else:
                                match_consecutive = False
                            if match_consecutive:
                                break_loop = True
                                break
                        else:
                            if next_dp == dp_candidate or next_dq == dq_candidate:
                                ignore = 1
                            else:
                                ignore = 2
                            match_consecutive = False
                            next_slice_candidates.append([p_candidate, q_candidate, dp_candidate, dq_candidate, ignore]) #キュー上優先順位を低くする

                    if len(next_slice_candidates)==16:
                        break_loop = True
                        break
                if break_loop:
                    break
            if break_loop:
                break
        if break_loop:
            break

    if match_consecutive:
        next_slice_candidates = [next_slice_candidates[0]]

    for next_slice in next_slice_candidates:
        next_slice.append(next_bit+3) #bit
        next_slice.append([*parameter_so_far(next_slice[0:4], next_bit+3, [p_so_far, q_so_far, dp_so_far, dq_so_far], tkp, tkq)])
        next_slice.append(kp) #kp
        next_slice.append(kq) #kq
        next_slice.append(copy.deepcopy(ignore_chain))
        next_slice.append(score)
    return next_slice_candidates

def HeningerShacham(hypothetical_dp, hypothetical_dq, N, e, k_cand, correct_kp, correct_kq, ki, num_error):
    global eval_count
    global start_time
    global return_time
    start_time = time.monotonic()
    dp_bin = transform_to_bit_list(hypothetical_dp)
    dq_bin = transform_to_bit_list(hypothetical_dq)
    dp_bin_0pad = [0 for _ in range(16)] #kp < 65536 -> tkp < 16
    dq_bin_0pad = [0 for _ in range(16)]
    dp_bin_0pad.extend(dp_bin)
    dq_bin_0pad.extend(dq_bin)

    node_queue = []
    count = 0
    eval_count = 0

    node_list_sub = Parallel(n_jobs=1, verbose=0)(delayed(generate_initial_node_for_kpair)(k_pair, dp_bin_0pad, dq_bin_0pad, candidates) for k_pair in tqdm(k_cand))
    for node_list in node_list_sub:
        for node in node_list:
            heapq.heappush(node_queue, (node[4], copy.deepcopy(node)))
    already_checked = False
    while len(node_queue) > 0:
        num_worker = 1
        if node_queue[0][-1][5] < 512:
            num_worker = 1
            (score, current_node) = heapq.heappop(node_queue)#[p,q,dp,dq,ignore,bit,[parameter_so_far],kp,kq,slice_chain]
            current_bit = current_node[5]
            ignore_chain = current_node[9]
            ignoring_estimation_count = sum(ignore_chain)
            next_node_list = eval_next_node(current_node, dp_bin_0pad, dq_bin_0pad, dp_candidates_list, dq_candidates_list, score)
            for next_node in next_node_list:
                score = next_node.pop()
                current_bit = next_node[5]
                ignore_chain = next_node[9] #[p,q,dp,dq,ignore,bit,[parameter_so_far],kp,kq,sc]
                ignoring_estimation_count = sum(ignore_chain)
                ignore = next_node[4]+ignoring_estimation_count
                
                match_consecutive_more = (current_bit >= 12) and (next_node[4] == 0) and (ignore_chain[-1] +  ignore_chain[-2] == 0)
                match_consecutive_extremely = (current_bit >= 16) and (next_node[4] == 0) and (ignore_chain[-3] + ignore_chain[-2] + ignore_chain[-1] == 0)
                
                might_be_ignoring_prediction = (current_bit >= 16) and (next_node[4] == 2) and (ignore_chain[-3] + ignore_chain[-2] + ignore_chain[-1] == 6)

                if match_consecutive:
                    cost = score - 1
                if match_consecutive_extremely:
                    cost -= 10
                elif might_be_ignoring_prediction:
                    cost = score + 10
                else:
                    cost = score + next_node[4]

                heapq.heappush(node_queue, (cost, copy.deepcopy(next_node)))
                node_queue.sort(key=lambda x: x[0])
        else:
            (score, current_node) = heapq.heappop(node_queue)
            #[p,q,dp,dq,ignore,bit,[parameter_so_far],kp,kq,slice_chain]
            ignore_chain = current_node.pop() #[p,q,dp,dq,ignore,bit,[parameter_so_far],kp,kq]
            kq = current_node.pop() #[p,q,dp,dq,ignore,bit,[parameter_so_far],kp]
            kp = current_node.pop() #[p,q,dp,dq,ignore,bit,[parameter_so_far]]
            parameter_so_far_list = current_node.pop() #[p,q,dp,dq,ignore,bit]
            current_bit = current_node.pop() #[p,q,dp,dq,ignore]
            current_ignore = current_node.pop() #[p,q,dp,dq]
            
            ignore_chain.append(current_ignore)
            ignoring_estimation_count = sum(ignore_chain)
            if current_bit == target_key_length:
                if result_is_ok(parameter_so_far_list, kp, kq):
                    return_time = time.monotonic()
                    elapsed_time = return_time-start_time
                    df_time = pd.read_csv("./result_time.csv",index_col=0)
                    df_branch = pd.read_csv("./result_branch.csv",index_col=0)
                    df_time.loc[ki][num_error] = elapsed_time
                    df_branch.loc[ki][num_error] = len(node_queue)
                    df_time.to_csv("./result_time.csv",float_format='%.4f')
                    df_branch.to_csv("./result_branch.csv")
                    (p_result, q_result, dp_result, dq_result) = (parameter_so_far_list[0], parameter_so_far_list[1], parameter_so_far_list[2], parameter_so_far_list[3])
                    return
                else:
                    continue


                
                

if __name__ == "__main__":

    with open("key_for_partial_key_exposure.txt") as f:
        reader = csv.reader(f)
        key_pool = [row for row in reader]

    num_error = 0 #change value as you want
    key_index = 0

    correct_p = key_pool[key_index][0]
    correct_q = key_pool[key_index][1]
    correct_dp = key_pool[key_index][2]
    correct_dq = key_pool[key_index][3]

    error_rate_list = [0, 0.01, 0.02, 0.03, 0.035, 0.04, 0.05, 0.06, 0.07, 0.075, 0.08]
    error_rate = error_rate_list[num_error]

    N = int(correct_p, 16) * int(correct_q, 16)
    e = 0x010001

    correct_kp = int((e*int(correct_dp, 16) - 1)/(int(correct_p, 16)-1))
    correct_kq = int((e*int(correct_dq, 16) - 1)/(int(correct_q, 16)-1))
    k_cand = [[correct_kp,correct_kq]]
    est_dp = errored_d(correct_dp, None, error_rate) #give errors to key value 
    est_dq = errored_d(correct_dq, None, error_rate)
    HeningerShacham(est_dp, est_dq, N, e, k_cand, correct_kp, correct_kq, key_index, num_error)

