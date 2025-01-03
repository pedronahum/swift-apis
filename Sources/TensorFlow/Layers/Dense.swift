import _Differentiation

/// A densely-connected neural network layer.
///
/// `Dense` implements the operation `activation(matmul(input, weight) + bias)`, where `weight` is
/// a weight matrix, `bias` is a bias vector, and `activation` is an element-wise activation
/// function.
///
/// This layer also supports 3-D weight tensors with 2-D bias matrices. In this case the first
/// dimension of both is treated as the batch size that is aligned with the first dimension of
/// `input` and the batch variant of the `matmul(_:_:)` operation is used, thus using a different
/// weight and bias for each element in the input batch.
@frozen
public struct Dense<Scalar: TensorFlowFloatingPoint>: Layer {
  // Mark conformance to `Layer`:
  public typealias Input = Tensor<Scalar>
  public typealias Output = Tensor<Scalar>

  /// The weight matrix.
  public var weight: Tensor<Scalar>
  /// The bias vector.
  public var bias: Tensor<Scalar>
  /// The element-wise activation function.
  @noDerivative public let activation: Activation
  /// Indicates whether this is a batched dense layer (weight is rank-3).
  @noDerivative internal let batched: Bool
  /// Whether or not a bias is used. (Because we can't store optional differentiable property easily.)
  @noDerivative private let useBias: Bool

  /// The element-wise activation function type.
  public typealias Activation = @differentiable(reverse) (Tensor<Scalar>) -> Tensor<Scalar>

  // -------------------------------------
  // MARK: - Initializers

  /// Creates a `Dense` layer from the given weight, optional bias, and activation function.
  ///
  /// - Note: `weight` is the only explicitly differentiable parameter here. `bias` is also
  ///   differentiable so you can use them both in training.
  @differentiable(reverse)
  public init(
    weight: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation
  ) {
    precondition(weight.rank <= 3, "The rank of 'weight' must be <= 3.")
    precondition(
      bias == nil || bias!.rank <= 2,
      "The rank of 'bias' must be <= 2 if present."
    )
    self.weight = weight
    self.bias = bias ?? .zero
    self.activation = activation
    self.batched = (weight.rank == 3)
    self.useBias = (bias != nil)
  }

  // Custom derivative for the above init if you need it:
  // (Typically you can remove or keep it, depending on your usage.)
  @derivative(of: init, wrt: weight)
  @usableFromInline
  static func vjpInit(
    weight: Tensor<Scalar>,
    bias: Tensor<Scalar>? = nil,
    activation: @escaping Activation
  ) -> (value: Self, pullback: (TangentVector) -> Tensor<Scalar>) {
    let value = Dense(weight: weight, bias: bias, activation: activation)
    return (value, { v in v.weight })  // Only returning the weight portion.
  }

  /// Creates a `Dense` layer with the specified input size, output size, and element-wise
  /// activation function. The weight matrix is created with shape `[inputSize, outputSize]` and
  /// the bias vector is created with shape `[outputSize]`.
  ///
  /// - Parameters:
  ///   - inputSize: The dimensionality of the input space.
  ///   - outputSize: The dimensionality of the output space.
  ///   - activation: The activation function to use. Default is `identity(_:)`.
  ///   - useBias: Whether to include a bias term. Default `true`.
  ///   - weightInitializer: Initializer for `weight`.
  ///   - biasInitializer: Initializer for `bias`.
  public init(
    inputSize: Int,
    outputSize: Int,
    activation: @escaping Activation = identity,
    useBias: Bool = true,
    weightInitializer: ParameterInitializer<Scalar> = glorotUniform(),
    biasInitializer: ParameterInitializer<Scalar> = zeros()
  ) {
    self.init(
      weight: weightInitializer([inputSize, outputSize]),
      bias: useBias ? biasInitializer([outputSize]) : nil,
      activation: activation
    )
  }

  // -------------------------------------
  // MARK: - Forward & callAsFunction

  /// Returns the output obtained from applying the layer to the given input.
  /// - Parameter input: The input to the layer.
  /// - Returns: The output of shape `[batch size, outputSize]`.
  @differentiable(reverse, wrt: (self, input))
  public func forward(_ input: Tensor<Scalar>) -> Tensor<Scalar> {
    if batched {
      // weight is rank-3 => we do a batch matmul for each sub-batch in dimension 0
      // Input shape: [batchSize, inputFeatures]
      // weight shape: [batchSize, inputFeatures, outputFeatures]
      // => batch matmul => result shape: [batchSize, outputFeatures]
      let hidden = matmul(input.expandingShape(at: 1), weight).squeezingShape(at: 1)
      return activation(useBias ? hidden + bias : hidden)
    } else {
      // standard dense: weight shape [inFeatures, outFeatures], bias shape [outFeatures].
      let logits = matmul(input, weight)
      return activation(useBias ? (logits + bias) : logits)
    }
  }

  /// Satisfies the `Layer` requirement. Delegates to `forward(_:)`.
  @differentiable(reverse, wrt: (self, input))
  public func callAsFunction(_ input: Input) -> Output {
    forward(input)
  }
}
