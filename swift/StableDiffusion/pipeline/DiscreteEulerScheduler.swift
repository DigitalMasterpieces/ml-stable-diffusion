import Accelerate
import CoreML


/// Prediction type of the scheduler function
@available(iOS 16.2, macOS 13.1, *)
public enum PredictionType: String {
    /// Predicting the noise of the diffusion process
    case epsilon
    /// Directly predicts the noisy sample
    case sample
    /// See section 2.4 https://imagen.research.google/video/paper.pdf
    case vPrediction = "v_prediction"
}

/// A first-order discrete Euler scheduler.
///
/// Matches the behavior of Hugging Face `EulerDiscreteScheduler`.
/// See: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py
///
/// This scheduler assumes the model predicts noise (`epsilon`).
///
/// Update rule:
/// ```
/// x0 = (x_t - sigma_t * eps_t) / alpha_t
/// d_sample = (x_t - x0) / sigma_t
/// x_{t-Δt} = x_t + (sigma_{t-Δt} - sigma_t) * d_sample
/// ```
@available(iOS 16.2, macOS 13.1, *)
public class DiscreteEulerScheduler: Scheduler {
    public var timeSteps: [Double]

    public let trainStepCount: Int
        public let inferenceStepCount: Int
        public let betas: [Float]
        public let alphas: [Float]
        public let alphasCumProd: [Float]
        public let sigmas: [Double]
    public let predictionType: PredictionType
        public let initNoiseSigma: Float

        public private(set) var modelOutputs: [MLShapedArray<Float32>] = []


    public init(strength: Float? = nil,
                stepCount: Int = 50,
                trainStepCount: Int = 1000,
                betaSchedule: BetaSchedule = .scaledLinear,
                betaStart: Float = 0.00085,
                betaEnd: Float = 0.012,
                stepsOffset: Int? = nil,
                predictionType: PredictionType = .epsilon,
                timestepSpacing: TimeStepSpacing? = nil,
                useKarrasSigmas: Bool = false) {

        self.trainStepCount = trainStepCount
        self.inferenceStepCount = stepCount
        self.predictionType = predictionType
        let timestepSpacing = timestepSpacing ?? .linspace

        switch betaSchedule {
            case .linear:
                self.betas = linspace(betaStart, betaEnd, trainStepCount)
            case .scaledLinear:
                self.betas = linspace(pow(betaStart, 0.5), pow(betaEnd, 0.5), trainStepCount).map({ $0 * $0 })
            }

        self.alphas = betas.map({ 1.0 - $0 })
        var alphasCumProd = self.alphas
        for i in 1..<alphasCumProd.count {
            alphasCumProd[i] *= alphasCumProd[i -  1]
        }
        self.alphasCumProd = alphasCumProd

        var timeSteps: [Double]
        switch timestepSpacing {
            case .linspace, .karras:
                timeSteps = linspaceD(0, Double(trainStepCount - 1), stepCount).reversed()
            case .leading:
                let stepRatio = trainStepCount / stepCount
                timeSteps = (0..<stepCount).map { Double($0 * stepRatio) + Double(stepsOffset ?? 0) }.reversed()
            case .trailing:
                let stepRatio = Double(trainStepCount) / Double(stepCount)
                timeSteps = stride(from: Double(trainStepCount), to: 1, by: -stepRatio).map { round($0) - 1 }
        }

        var sigmas: [Double] = alphasCumProd.map { Double(pow((1 - $0) / $0, 0.5)) }
        sigmas = vDSP.linearInterpolate(elementsOf: sigmas, using: timeSteps) + [0]

        switch timestepSpacing {
            case .linspace, .leading, .karras:
                self.initNoiseSigma = Float(sigmas.max() ?? 1)
            case .trailing:
                self.initNoiseSigma = pow(pow(Float(sigmas.max() ?? 1), 2) + 1, 0.5)
        }

        if let strength {
            let initTimestep = min(Int(Float(stepCount) * strength), stepCount)
            let tStart = min(timeSteps.count - 1, max(stepCount - initTimestep, 0))
            timeSteps = Array(timeSteps[tStart..<timeSteps.count])
            sigmas = Array(sigmas[tStart..<sigmas.count])
        }
        self.timeSteps = timeSteps
        self.sigmas = sigmas
    }

    public func scaleModelInput(timeStep t: Double, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
            let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
            let sigma = Float32(sigmas[stepIndex])
            let scale: Float32 = pow(pow(sigma, 2) + 1, 0.5)
            return MLShapedArray(unsafeUninitializedShape: sample.shape) { scalars, _ in
                sample.withUnsafeShapedBufferPointer { sample, _, _ in
                    vDSP.divide(sample, scale, result: &scalars)
                }
            }
        }



    /// Calculate timesteps optionally applying image-to-image strength
    /*
    public func calculateTimesteps(strength: Float?) -> [Int] {
        guard let strength = strength else { return timeSteps }
        let startStep = Int(Float(timeSteps.count) * strength)
        return Array(timesteps[startStep..<timesteps.count])
    }
     */

    /// Perform a single Euler step
    public func step(output: MLShapedArray<Float32>, timeStep t: Double, sample: MLShapedArray<Float32>) -> MLShapedArray<Float32> {
        let scalarCount = sample.scalarCount

                let stepIndex = timeSteps.firstIndex(of: t) ?? timeSteps.count - 1
                let sigma = Float32(sigmas[stepIndex])

                // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
                let predOriginalSample: MLShapedArray<Float32>
                switch predictionType {
                case .epsilon:
                    predOriginalSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                        sample.withUnsafeShapedBufferPointer { sample, _, _ in
                            output.withUnsafeShapedBufferPointer { output, _, _ in
                                for i in 0..<scalarCount {
                                    scalars.initializeElement(at: i, to: sample[i] - sigma * output[i])
                                }
                            }
                        }
                    }
                case .sample:
                    predOriginalSample = output
                case .vPrediction:
                    // * c_out + input * c_skip
                    let sigmaPow: Float32 = pow(sigma, 2) + 1
                    let sigmaAux: Float32 = -sigma / pow(sigmaPow, 0.5)
                    predOriginalSample = MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                        sample.withUnsafeShapedBufferPointer { sample, _, _ in
                            output.withUnsafeShapedBufferPointer { output, _, _ in
                                for i in 0..<scalarCount {
                                    scalars.initializeElement(at: i, to: output[i] * sigmaAux + (sample[i] / sigmaPow))
                                }
                            }
                        }
                    }
                }

                modelOutputs.removeAll(keepingCapacity: true)
                modelOutputs.append(predOriginalSample)

                // 2. Convert to an ODE derivative
                let dt: Float32 = Float32(sigmas[stepIndex + 1]) - sigma
                return MLShapedArray(unsafeUninitializedShape: output.shape) { scalars, _ in
                    sample.withUnsafeShapedBufferPointer { sample, _, _ in
                        predOriginalSample.withUnsafeShapedBufferPointer { original, _, _ in
                            for i in 0..<scalarCount {
                                let derivative = (sample[i] - original[i]) / sigma
                                scalars.initializeElement(at: i, to: sample[i] + derivative * dt)
                            }
                        }
                    }
                }
    }

    public func addNoise(
            originalSample: MLShapedArray<Float32>,
            noise: [MLShapedArray<Float32>],
            timeStep t: Double?
        ) -> [MLShapedArray<Float32>] {
            let stepIndex = t.flatMap { timeSteps.firstIndex(of: $0) } ?? 0
            let sigma = Float32(sigmas[stepIndex])
            let noisySamples = noise.map { noise in
                MLShapedArray(unsafeUninitializedShape: originalSample.shape) { scalars, _ in
                    originalSample.withUnsafeShapedBufferPointer { sample, _, _ in
                        noise.withUnsafeShapedBufferPointer { noise, _, _ in
                            for i in 0..<originalSample.scalarCount {
                                scalars.initializeElement(at: i, to: sample[i] + noise[i] * sigma)
                            }
                        }
                    }
                }
            }
            return noisySamples
        }
}
