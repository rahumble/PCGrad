from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import random

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer

from TaskWeighter import compute_task_weightings

GATE_OP = 1


class PCGrad(optimizer.Optimizer):
    '''Tensorflow implementation of PCGrad.
    Gradient Surgery for Multi-Task Learning: https://arxiv.org/pdf/2001.06782.pdf
    '''

    def __init__(self, optimizer, use_locking=False, name="PCGrad", type="PCGrad"):
        """optimizer: the optimizer being wrapped
        """
        super(PCGrad, self).__init__(use_locking, name)
        self.optimizer = optimizer
        self.type = type

    def compute_gradients(self, loss, var_list=None,
                        gate_gradients=GATE_OP,
                        aggregation_method=None,
                        colocate_gradients_with_ops=False,
                        grad_loss=None):
        assert type(loss) is list
        num_tasks = len(loss)

        loss = tf.stack(loss)
        tf.random.shuffle(loss)

        # Compute per-task gradients.
        grads_task = tf.vectorized_map(lambda x: tf.concat([tf.reshape(grad, [-1,]) 
                            for grad in tf.gradients(x, var_list) 
                            if grad is not None], axis=0), loss)

        if self.type == "PCGrad":
            # Compute gradient projections.
            def proj_grad(grad_task):
                for k in range(num_tasks):
                    inner_product = tf.reduce_sum(grad_task*grads_task[k])
                    proj_direction = inner_product / tf.reduce_sum(grads_task[k]*grads_task[k])
                    grad_task = grad_task - tf.minimum(proj_direction, 0.) * grads_task[k]
                return grad_task

            proj_grads_flatten = tf.vectorized_map(proj_grad, grads_task)

            # Unpack flattened projected gradients back to their original shapes.
            proj_grads = []
            for j in range(num_tasks):
                start_idx = 0
                for idx, var in enumerate(var_list):
                    grad_shape = var.get_shape()
                    flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                    proj_grad = proj_grads_flatten[j][start_idx:start_idx+flatten_dim]
                    proj_grad = tf.reshape(proj_grad, grad_shape)
                    if len(proj_grads) < len(var_list):
                        proj_grads.append(proj_grad)
                    else:
                        proj_grads[idx] += proj_grad               
                    start_idx += flatten_dim
            grads_and_vars = list(zip(proj_grads, var_list))
            return grads_and_vars
        else:
            # Build the gradient inner product matrix M
            M = np.zeros((num_tasks, num_tasks))
            for i in range(num_tasks):
                for j in range(i, num_tasks):
                    M[i,j] = tf.reduce_sum(grads_task[i] * grads_task[j]).numpy()
                    M[j,i] = M[i,j]

            # Recover task weightings
            weights = compute_task_weightings(self.type, M)

            # Unpack flattened gradients back to their original shapes. Scale by task weighting.
            proj_grads = []
            for j in range(num_tasks):
                start_idx = 0
                for idx, var in enumerate(var_list):
                    grad_shape = var.get_shape()
                    flatten_dim = np.prod([grad_shape.dims[i].value for i in range(len(grad_shape.dims))])
                    proj_grad = weights[j] * grads_task[j][start_idx:start_idx+flatten_dim]
                    proj_grad = tf.reshape(proj_grad, grad_shape)
                    if len(proj_grads) < len(var_list):
                        proj_grads.append(proj_grad)
                    else:
                        proj_grads[idx] += proj_grad
                    start_idx += flatten_dim
            grads_and_vars = list(zip(proj_grads, var_list))
            return grads_and_vars

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

    def _prepare(self):
        self.optimizer._prepare()

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        return self.optimizer._finish(update_ops, name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param
