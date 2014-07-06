import numpy as np

n_arms = 6

A = [[ 0.5         ,1.          ,2.03181837  ,1.          ,1.          ,0.05792985],
     [ 1.          ,0.5         ,1.05872186  ,1.          ,1.          ,1.        ],
     [ 1.03181837  ,0.05872186  ,0.5         ,0.05876946  ,0.05820571  ,1.        ],
     [ 1.          ,1.          ,1.05876946  ,0.5         ,1.          ,1.        ],
     [ 1.          ,1.          ,1.05820571  ,1.          ,0.5         ,1.        ],
     [ 1.05792985  ,1.          ,1.          ,1.          ,1.          ,0.5       ]]

half_matrix = np.ones([n_arms, n_arms])*0.5

greater_then_half_matrix = A >= half_matrix

c_vector = [False]*n_arms

for left_arm in range(n_arms):

            if np.all(greater_then_half_matrix[left_arm, :]):
                c_vector[left_arm] = True
            else:
                c_vector[left_arm] = False

print c_vector