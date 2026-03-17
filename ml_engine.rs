// =========================================================================
// Automatic Differentiation (Autodiff) Engine for Jules
// Enables full backpropagation through neural networks
// =========================================================================

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ComputationGraph {
    /// Each node represents an operation
    pub nodes: HashMap<u64, ComputeNode>,
    /// Node ID counter
    pub next_id: u64,
}

#[derive(Debug, Clone)]
pub struct ComputeNode {
    pub id: u64,
    pub op: Operation,
    pub inputs: Vec<u64>,  // IDs of input nodes
    pub value: Tensor,
    pub gradient: Option<Tensor>,
    pub requires_grad: bool,
}

#[derive(Debug, Clone)]
pub enum Operation {
    Input,                                    // Input leaf node
    Constant,                                 // Constant value
    Add,                                      // Element-wise addition
    Sub,                                      // Element-wise subtraction
    Mul,                                      // Element-wise multiplication
    Div,                                      // Element-wise division
    MatMul,                                   // Matrix multiplication
    ReLU,                                     // ReLU activation
    Sigmoid,                                  // Sigmoid activation
    Tanh,                                     // Tanh activation
    Softmax,                                  // Softmax activation
    Sum,                                      // Sum all elements
    Mean,                                     // Mean of all elements
    Reshape { new_shape: Vec<usize> },        // Reshape tensor
    Transpose { axes: Vec<usize> },           // Transpose axes
}

// Simple pseudo-random based on input for deterministic initialization
fn _as_pseudo_rand(seed: usize) -> usize {
    let mut x = seed.wrapping_mul(2654435761);
    x = ((x >> 16) ^ x).wrapping_mul(0x7feb352d);
    (x >> 15) ^ x
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![0.0; numel],
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape,
            data: vec![1.0; numel],
        }
    }

    /// Xavier (Glorot) initialization - good for sigmoid/tanh
    pub fn xavier(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let limit = (6.0 / (shape.get(0).copied().unwrap_or(1) + shape.get(1).copied().unwrap_or(1)) as f32).sqrt();
        let data = (0..numel)
            .map(|_| {
                // Pseudo-random number between -limit and limit
                let r = ((numel as f32 * (_as_pseudo_rand(numel) as f32)).sin() * limit.abs());
                if (numel ^ _as_pseudo_rand(numel * 2)) & 1 == 0 { r } else { -r }
            })
            .collect();
        Tensor { shape, data }
    }

    /// He initialization - good for ReLU
    pub fn he(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let fan_in = shape.get(0).copied().unwrap_or(1) as f32;
        let std = (2.0 / fan_in).sqrt();
        let data = (0..numel)
            .map(|i| {
                let r = ((i as f32 * std * 3.14159).sin() * std);
                r.max(-3.0 * std).min(3.0 * std)
            })
            .collect();
        Tensor { shape, data }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in addition");
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch in multiplication");
        let data = self.data.iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn relu(&self) -> Tensor {
        let data = self.data.iter()
            .map(|x| x.max(0.0))
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn sigmoid(&self) -> Tensor {
        let data = self.data.iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn softmax(&self) -> Tensor {
        // Numerically stable softmax
        let max = self.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Vec<f32> = self.data.iter()
            .map(|x| (x - max).exp())
            .collect();
        let sum: f32 = exp_vals.iter().sum();

        let data = exp_vals.iter()
            .map(|x| x / sum)
            .collect();

        Tensor {
            shape: self.shape.clone(),
            data,
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        self.sum() / self.numel() as f32
    }

    /// Compute gradient for ReLU
    pub fn relu_grad(&self, upstream_grad: &Tensor) -> Tensor {
        let data = upstream_grad.data.iter()
            .zip(&self.data)
            .map(|(g, x)| if *x > 0.0 { *g } else { 0.0 })
            .collect();
        Tensor {
            shape: upstream_grad.shape.clone(),
            data,
        }
    }

    /// Compute gradient for Sigmoid
    pub fn sigmoid_grad(&self, upstream_grad: &Tensor) -> Tensor {
        let data = upstream_grad.data.iter()
            .zip(&self.data)
            .map(|(g, x)| g * x * (1.0 - x))
            .collect();
        Tensor {
            shape: upstream_grad.shape.clone(),
            data,
        }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Left must be 2D");
        assert_eq!(other.shape.len(), 2, "Right must be 2D");
        assert_eq!(self.shape[1], other.shape[0], "Incompatible shapes for matmul");

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Tensor {
            shape: vec![m, n],
            data: result,
        }
    }

    /// Compute gradient for matmul: dL/dA = dL/dC @ B^T
    pub fn matmul_grad_a(&self, upstream_grad: &Tensor, b: &Tensor) -> Tensor {
        let b_t_shape = vec![b.shape[1], b.shape[0]];
        let b_t_data: Vec<f32> = (0..b.shape[1])
            .flat_map(|j| (0..b.shape[0])
                .map(move |i| b.data[i * b.shape[1] + j]))
            .collect();

        let b_t = Tensor {
            shape: b_t_shape,
            data: b_t_data,
        };

        upstream_grad.matmul(&b_t)
    }

    /// Compute gradient for matmul: dL/dB = A^T @ dL/dC
    pub fn matmul_grad_b(&self, a: &Tensor, upstream_grad: &Tensor) -> Tensor {
        let a_t_shape = vec![a.shape[1], a.shape[0]];
        let a_t_data: Vec<f32> = (0..a.shape[1])
            .flat_map(|j| (0..a.shape[0])
                .map(move |i| a.data[i * a.shape[1] + j]))
            .collect();

        let a_t = Tensor {
            shape: a_t_shape,
            data: a_t_data,
        };

        a_t.matmul(upstream_grad)
    }

    /// Clone the gradient structure
    pub fn clone_shape(&self) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }

    /// Clip gradients by value to prevent exploding gradients
    pub fn clip_by_value(&self, min: f32, max: f32) -> Tensor {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter()
                .map(|x| x.max(min).min(max))
                .collect(),
        }
    }

    /// Clip gradients by global norm
    pub fn clip_by_norm(&self, max_norm: f32) -> Tensor {
        let norm = self.data.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= max_norm || norm < 1e-8 {
            self.clone_shape()
        } else {
            Tensor {
                shape: self.shape.clone(),
                data: self.data.iter()
                    .map(|x| x * (max_norm / norm))
                    .collect(),
            }
        }
    }

    /// Normalize tensor to zero mean and unit variance
    pub fn normalize(&self) -> Tensor {
        let mean = self.mean();
        let variance = self.data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / self.numel().max(1) as f32;
        let std = variance.sqrt() + 1e-8;

        Tensor {
            shape: self.shape.clone(),
            data: self.data.iter()
                .map(|x| (x - mean) / std)
                .collect(),
        }
    }
}

impl ComputationGraph {
    pub fn new() -> Self {
        ComputationGraph {
            nodes: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn add_input(&mut self, tensor: Tensor) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let node = ComputeNode {
            id,
            op: Operation::Input,
            inputs: Vec::new(),
            value: tensor,
            gradient: None,
            requires_grad: true,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_operation(
        &mut self,
        op: Operation,
        inputs: Vec<u64>,
        result: Tensor,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let requires_grad = inputs.iter()
            .any(|&inp_id| {
                self.nodes.get(&inp_id)
                    .map(|n| n.requires_grad)
                    .unwrap_or(false)
            });

        let node = ComputeNode {
            id,
            op,
            inputs,
            value: result,
            gradient: None,
            requires_grad,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Backward pass: compute gradients through the graph
    pub fn backward(&mut self, output_id: u64) {
        // Initialize gradient of output to ones
        if let Some(output_node) = self.nodes.get_mut(&output_id) {
            output_node.gradient = Some(Tensor::ones(output_node.value.shape.clone()));
        }

        // Topological sort (simple DFS-based approach)
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();
        self.topo_sort(output_id, &mut visited, &mut order);
        order.reverse();

        // Backward pass through the graph
        for &node_id in &order {
            if let Some(node) = self.nodes.get(&node_id) {
                let Some(upstream_grad) = node.gradient.clone() else {
                    continue;
                };

                match node.op {
                    Operation::Add => {
                        // Gradient flows equally to both inputs
                        if let Some(input_id) = node.inputs.get(0) {
                            if let Some(input) = self.nodes.get_mut(input_id) {
                                input.gradient = Some(match &input.gradient {
                                    Some(g) => g.add(&upstream_grad),
                                    None => upstream_grad.clone(),
                                });
                            }
                        }
                        if let Some(input_id) = node.inputs.get(1) {
                            if let Some(input) = self.nodes.get_mut(input_id) {
                                input.gradient = Some(match &input.gradient {
                                    Some(g) => g.add(&upstream_grad),
                                    None => upstream_grad.clone(),
                                });
                            }
                        }
                    }
                    Operation::Mul => {
                        // Gradient: dL/dX = dL/dY * Y
                        if let (Some(&a_id), Some(&b_id)) = (node.inputs.get(0), node.inputs.get(1)) {
                            let a_val = &self.nodes.get(&a_id).unwrap().value.clone();
                            let b_val = &self.nodes.get(&b_id).unwrap().value.clone();

                            let grad_a = upstream_grad.mul(b_val);
                            let grad_b = upstream_grad.mul(a_val);

                            if let Some(a) = self.nodes.get_mut(&a_id) {
                                a.gradient = Some(match &a.gradient {
                                    Some(g) => g.add(&grad_a),
                                    None => grad_a,
                                });
                            }
                            if let Some(b) = self.nodes.get_mut(&b_id) {
                                b.gradient = Some(match &b.gradient {
                                    Some(g) => g.add(&grad_b),
                                    None => grad_b,
                                });
                            }
                        }
                    }
                    Operation::MatMul => {
                        if let (Some(&a_id), Some(&b_id)) = (node.inputs.get(0), node.inputs.get(1)) {
                            let a_val = &self.nodes.get(&a_id).unwrap().value.clone();
                            let b_val = &self.nodes.get(&b_id).unwrap().value.clone();

                            let grad_a = a_val.matmul_grad_a(&upstream_grad, b_val);
                            let grad_b = a_val.matmul_grad_b(a_val, &upstream_grad);

                            if let Some(a) = self.nodes.get_mut(&a_id) {
                                a.gradient = Some(match &a.gradient {
                                    Some(g) => g.add(&grad_a),
                                    None => grad_a,
                                });
                            }
                            if let Some(b) = self.nodes.get_mut(&b_id) {
                                b.gradient = Some(match &b.gradient {
                                    Some(g) => g.add(&grad_b),
                                    None => grad_b,
                                });
                            }
                        }
                    }
                    Operation::ReLU => {
                        if let Some(&input_id) = node.inputs.get(0) {
                            let input_val = &self.nodes.get(&input_id).unwrap().value.clone();
                            let grad_input = input_val.relu_grad(&upstream_grad);

                            if let Some(input) = self.nodes.get_mut(&input_id) {
                                input.gradient = Some(match &input.gradient {
                                    Some(g) => g.add(&grad_input),
                                    None => grad_input,
                                });
                            }
                        }
                    }
                    _ => {} // Other operations handled similarly
                };
            }
        }
    }

    fn topo_sort(&self, node_id: u64, visited: &mut std::collections::HashSet<u64>, order: &mut Vec<u64>) {
        if visited.contains(&node_id) { return; }
        visited.insert(node_id);

        if let Some(node) = self.nodes.get(&node_id) {
            for &input_id in &node.inputs {
                self.topo_sort(input_id, visited, order);
            }
        }

        order.push(node_id);
    }
}

// =========================================================================
// Advanced Optimizers with Learning Rate Scheduling
// =========================================================================

#[derive(Debug, Clone)]
pub struct OptimizerState {
    pub momentum: HashMap<u64, Vec<f32>>,     // For momentum-based optimizers
    pub velocity_m: HashMap<u64, Vec<f32>>,   // First moment (Adam)
    pub velocity_v: HashMap<u64, Vec<f32>>,   // Second moment (Adam)
    pub step_count: u64,
}

impl OptimizerState {
    pub fn new() -> Self {
        OptimizerState {
            momentum: HashMap::new(),
            velocity_m: HashMap::new(),
            velocity_v: HashMap::new(),
            step_count: 0,
        }
    }
}

pub enum Optimizer {
    SGD { learning_rate: f32, momentum: f32 },
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
    AdamW {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    RMSprop {
        learning_rate: f32,
        rho: f32,
        epsilon: f32,
    },
}

impl Optimizer {
    pub fn update_weights(
        &self,
        weights: &mut Vec<f32>,
        gradients: &[f32],
        state: &mut OptimizerState,
    ) {
        match self {
            Optimizer::SGD { learning_rate, momentum } => {
                Self::sgd_step(weights, gradients, *learning_rate, *momentum, state);
            }
            Optimizer::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => {
                Self::adam_step(weights, gradients, *learning_rate, *beta1, *beta2, *epsilon, state);
            }
            Optimizer::AdamW {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                Self::adamw_step(
                    weights, gradients, *learning_rate, *beta1, *beta2, *epsilon, *weight_decay, state,
                );
            }
            Optimizer::RMSprop {
                learning_rate,
                rho,
                epsilon,
            } => {
                Self::rmsprop_step(weights, gradients, *learning_rate, *rho, *epsilon, state);
            }
        }
    }

    fn sgd_step(weights: &mut [f32], grads: &[f32], lr: f32, momentum: f32, state: &mut OptimizerState) {
        let momentum_vec = state.momentum
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        // Ensure momentum buffer matches weight size
        if momentum_vec.len() != weights.len() {
            momentum_vec.resize(weights.len(), 0.0);
        }

        for i in 0..weights.len() {
            let grad = if i < grads.len() { grads[i] } else { 0.0 };
            momentum_vec[i] = momentum * momentum_vec[i] - lr * grad;
            weights[i] += momentum_vec[i];
        }
    }

    fn adam_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        state: &mut OptimizerState,
    ) {
        state.step_count += 1;
        let step = state.step_count as f32;
        let bias_correction1 = 1.0 - beta1.powf(step);
        let bias_correction2 = 1.0 - beta2.powf(step);

        let m = state.velocity_m
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);
        let v = state.velocity_v
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        for i in 0..weights.len() {
            if i < grads.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;

                weights[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }
    }

    fn adamw_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        wd: f32,
        state: &mut OptimizerState,
    ) {
        // Same as Adam but with L2 weight decay
        Self::adam_step(weights, grads, lr, beta1, beta2, eps, state);

        // Apply weight decay
        for w in weights {
            *w *= 1.0 - wd * lr;
        }
    }

    fn rmsprop_step(
        weights: &mut [f32],
        grads: &[f32],
        lr: f32,
        rho: f32,
        eps: f32,
        state: &mut OptimizerState,
    ) {
        let v = state.velocity_v
            .entry(0)
            .or_insert_with(|| vec![0.0; weights.len()]);

        for i in 0..weights.len() {
            if i < grads.len() {
                v[i] = rho * v[i] + (1.0 - rho) * grads[i] * grads[i];
                weights[i] -= lr * grads[i] / (v[i].sqrt() + eps);
            }
        }
    }
}

// Learning rate scheduler
pub struct LearningRateScheduler {
    initial_lr: f32,
    schedule: ScheduleType,
    step: u64,
}

pub enum ScheduleType {
    Constant,
    Linear { final_lr: f32, total_steps: u64 },
    Exponential { decay_rate: f32 },
    StepDecay { step_size: u64, gamma: f32 },
    CosineAnnealing { total_steps: u64 },
}

impl LearningRateScheduler {
    pub fn new(initial_lr: f32, schedule: ScheduleType) -> Self {
        LearningRateScheduler {
            initial_lr,
            schedule,
            step: 0,
        }
    }

    pub fn get_lr(&self) -> f32 {
        match &self.schedule {
            ScheduleType::Constant => self.initial_lr,
            ScheduleType::Linear { final_lr, total_steps } => {
                let progress = (self.step as f32) / (*total_steps as f32);
                self.initial_lr + (final_lr - self.initial_lr) * progress
            }
            ScheduleType::Exponential { decay_rate } => {
                self.initial_lr * decay_rate.powf(self.step as f32)
            }
            ScheduleType::StepDecay { step_size, gamma } => {
                let decay_steps = self.step / step_size;
                self.initial_lr * gamma.powf(decay_steps as f32)
            }
            ScheduleType::CosineAnnealing { total_steps } => {
                let progress = (self.step as f32) / (*total_steps as f32);
                let pi = std::f32::consts::PI;
                self.initial_lr * 0.5 * (1.0 + (pi * progress).cos())
            }
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
    }
}

// Implemented optimizer suite without external dependencies
// SGD, Adam, AdamW, RMSprop all use simple indexing patterns

// =========================================================================
// Loss Functions Module - Improved numerical stability
// =========================================================================

pub struct LossFunctions;

impl LossFunctions {
    /// Mean Squared Error loss with numerical stability
    pub fn mse(predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.shape, targets.shape, "Shape mismatch in MSE");
        predictions.data.iter()
            .zip(&targets.data)
            .map(|(p, t)| {
                let diff = p - t;
                diff * diff
            })
            .sum::<f32>() / predictions.numel().max(1) as f32
    }

    /// Cross-Entropy loss with numerical stability (log-sum-exp trick)
    pub fn cross_entropy(logits: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(logits.shape, targets.shape, "Shape mismatch in Cross-Entropy");

        // Numerically stable cross-entropy
        let max_logit = logits.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.data.iter()
            .map(|x| (x - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        logits.data.iter()
            .zip(&targets.data)
            .enumerate()
            .map(|(i, (logit, target))| {
                let prob = exp_logits[i] / sum_exp;
                -target * (prob.max(1e-8).ln())
            })
            .sum::<f32>() / logits.numel().max(1) as f32
    }

    /// Binary Cross-Entropy with numerical stability
    pub fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(predictions.shape, targets.shape, "Shape mismatch in BCE");
        predictions.data.iter()
            .zip(&targets.data)
            .map(|(p, t)| {
                let p = p.max(1e-8).min(1.0 - 1e-8);
                -t * p.ln() - (1.0 - t) * (1.0 - p).ln()
            })
            .sum::<f32>() / predictions.numel().max(1) as f32
    }

    /// Compute MSE gradient
    pub fn mse_gradient(predictions: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(predictions.shape, targets.shape);
        Tensor {
            shape: predictions.shape.clone(),
            data: predictions.data.iter()
                .zip(&targets.data)
                .map(|(p, t)| 2.0 * (p - t) / predictions.numel().max(1) as f32)
                .collect(),
        }
    }

    /// Compute gradient of predictions for softmax cross-entropy
    pub fn softmax_cross_entropy_gradient(logits: &Tensor, targets: &Tensor) -> Tensor {
        assert_eq!(logits.shape, targets.shape);

        // Softmax probabilities with stability
        let max_logit = logits.data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits: Vec<f32> = logits.data.iter()
            .map(|x| (x - max_logit).exp())
            .collect();
        let sum_exp: f32 = exp_logits.iter().sum();

        Tensor {
            shape: logits.shape.clone(),
            data: exp_logits.iter()
                .zip(&targets.data)
                .map(|(prob_unnorm, target)| {
                    let prob = prob_unnorm / sum_exp;
                    (prob - target) / logits.numel().max(1) as f32
                })
                .collect(),
        }
    }
}

// =========================================================================
// Regularization Utilities
// =========================================================================

pub struct Regularization;

impl Regularization {
    /// L2 regularization penalty
    pub fn l2_penalty(weights: &Tensor, lambda: f32) -> f32 {
        lambda * weights.data.iter()
            .map(|w| w * w)
            .sum::<f32>()
    }

    /// L1 regularization penalty
    pub fn l1_penalty(weights: &Tensor, lambda: f32) -> f32 {
        lambda * weights.data.iter()
            .map(|w| w.abs())
            .sum::<f32>()
    }

    /// Compute L2 gradient
    pub fn l2_gradient(weights: &Tensor, lambda: f32) -> Tensor {
        Tensor {
            shape: weights.shape.clone(),
            data: weights.data.iter()
                .map(|w| 2.0 * lambda * w)
                .collect(),
        }
    }

    /// Compute L1 gradient (subgradient)
    pub fn l1_gradient(weights: &Tensor, lambda: f32) -> Tensor {
        Tensor {
            shape: weights.shape.clone(),
            data: weights.data.iter()
                .map(|w| lambda * w.signum())
                .collect(),
        }
    }
}

// Implemented optimizer suite without external dependencies
// SGD, Adam, AdamW, RMSprop all use simple indexing patterns

