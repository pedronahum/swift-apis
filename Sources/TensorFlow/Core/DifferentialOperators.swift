// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ===------------------------------------------------------------------------------------------===//
// Free-function-style differential operators
// ===------------------------------------------------------------------------------------------===//

import _Differentiation

/// A helper function that returns rank-0 "1" with the same shape as `y`.
@inlinable
func rank0Pullback<Scalar: TensorFlowFloatingPoint, Result>(
  _ y: Tensor<Scalar>,
  pullback: @escaping (Tensor<Scalar>) -> Result
) -> Result
{
  precondition(y.rank == 0, "Return must be a rank-0 tensor.")
  let ones = Tensor<Scalar>(onesLike: y)
  return pullback(ones)
}

// A single-argument "value + gradient" function. 
@inlinable
public func s4tfValueWithGradient<T, Scalar>(
  at x: T,
  of f: @differentiable(reverse) (T) -> Tensor<Scalar>
) -> (value: Tensor<Scalar>, gradient: T.TangentVector)
where T: Differentiable, Scalar: TensorFlowFloatingPoint 
{
  // 1) Standard library function is spelled: valueWithPullback(at:of:)
  let (y, pullback) = valueWithPullback(at: x, of: f)
  // 2) Check rank-0, apply pullback(1).
  precondition(y.rank == 0, "Return must be rank-0 tensor.")
  let grad = rank0Pullback(y, pullback: pullback)
  return (y, grad)
}

@inlinable
public func s4tfGradient<T, Scalar>(
  at x: T,
  in f: @differentiable(reverse) (T) -> Tensor<Scalar>
) -> T.TangentVector
where T: Differentiable, Scalar: TensorFlowFloatingPoint
{
  s4tfValueWithGradient(at: x, of: f).gradient
}

// ----------------------------------------
//  Two-argument "value + gradient"
// ----------------------------------------
@inlinable
public func s4tfValueWithGradient<T, U, Scalar>(
  at x: T, _ y: U,
  in f: @differentiable(reverse) (T, U) -> Tensor<Scalar>
) -> (value: Tensor<Scalar>, gradient: (T.TangentVector, U.TangentVector))
where T: Differentiable, U: Differentiable, Scalar: TensorFlowFloatingPoint
{
  let (out, pullback) = valueWithPullback(at: x, y, of: f)
  precondition(out.rank == 0, "Return must be a scalar (rank-0) tensor.")
  let grads = pullbackOfOneLikeY(y: out, pullback: pullback)
  return (out, grads)
}

@inlinable
public func s4tfGradient<T, U, Scalar>(
  at x: T, _ y: U,
  in f: @differentiable(reverse) (T, U) -> Tensor<Scalar>
) -> (T.TangentVector, U.TangentVector)
where T: Differentiable, U: Differentiable, Scalar: TensorFlowFloatingPoint
{
  s4tfValueWithGradient(at: x, y, in: f).gradient
}

// ----------------------------------------
//  Three-argument "value + gradient"
// ----------------------------------------
@inlinable
public func s4tfValueWithGradient<T, U, V, Scalar>(
  at x: T, _ y: U, _ z: V,
  in f: @differentiable(reverse) (T, U, V) -> Tensor<Scalar>
) -> (value: Tensor<Scalar>, gradient: (T.TangentVector, U.TangentVector, V.TangentVector))
where T: Differentiable, U: Differentiable, V: Differentiable, Scalar: TensorFlowFloatingPoint
{
  let (out, pullback) = valueWithPullback(at: x, y, z, of: f)
  precondition(out.rank == 0, "Return must be a scalar (rank-0) tensor.")
  let grads = pullbackOfOneLikeY(y: out, pullback: pullback)
  return (out, grads)
}

@inlinable
public func s4tfGradient<T, U, V, Scalar>(
  at x: T, _ y: U, _ z: V,
  in f: @differentiable(reverse) (T, U, V) -> Tensor<Scalar>
) -> (T.TangentVector, U.TangentVector, V.TangentVector)
where T: Differentiable, U: Differentiable, V: Differentiable, Scalar: TensorFlowFloatingPoint
{
  s4tfValueWithGradient(at: x, y, z, in: f).gradient
}
