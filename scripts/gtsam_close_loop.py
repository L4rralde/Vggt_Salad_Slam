from time import perf_counter
import numpy as np
import gtsam #pip install gtsam-develop. Requires Python>=3.11

#Load predictions and relative transformations
root = '/Users/l4rralde/Desktop/calderon_output/'
measurements = np.load(f'{root}/measurements.npz')['arr_0']
estimations = np.load(f'{root}/estimated_poses.npz')['arr_0']

n_nodes = len(estimations)

graph = gtsam.NonlinearFactorGraph() #Instantiate graph
noise = 0.05*np.ones(15, dtype=float) #Default relative trasnformation/measurement noise
noise = gtsam.noiseModel.Diagonal.Sigmas(noise)
anchor_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-6] * 15) #Noise of the origin

#Add first Node: First pose. Anchor.
graph.add(
    gtsam.PriorFactorSL4(0, gtsam.SL4(), anchor_noise)
)

#Add all realitve measurements
for i, measure in enumerate(measurements[:-1]):
    print(i, i+1)
    graph.add(gtsam.BetweenFactorSL4(i, i+1, gtsam.SL4(measure), noise))
graph.add(
    gtsam.BetweenFactorSL4(
        n_nodes - 1,
        0,
        gtsam.SL4(measurements[-1]),
        noise
    )
)

#Instantiate initial values of the global poses to optimize
values = gtsam.Values()
global_pose = np.eye(4)
values.insert(0, gtsam.SL4(global_pose))
for i, meas in enumerate(measurements[:-1]):
    global_pose = global_pose @ meas
    values.insert(i+1, gtsam.SL4(global_pose))

#Non-Linear Least Square optimization Algorithm
params = gtsam.LevenbergMarquardtParams()
params.setVerbosityLM("SUMMARY") #Pass params only if you want to display stats
params.setVerbosity("ERROR")
#optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values, params)
optimizer = gtsam.LevenbergMarquardtOptimizer(graph, values)

initial_error = graph.error(values) #Error before optimization

start_opt = perf_counter()
result = optimizer.optimize()
end_opt = perf_counter()

final_error = graph.error(result) #Final error 

#Convert result to np ndarray in homogeneous form
gtsam_estimates = []
for i in range(len(estimations)):
    mat = result.atSL4(i).matrix()
    gtsam_estimates.append(mat/mat[-1, -1])
gtsam_estimates = np.asarray(gtsam_estimates)

#Save results
np.savez(f'{root}/gtsam_estimates.npz', gtsam_estimates)

print(initial_error)
print(final_error)
print(end_opt - start_opt)