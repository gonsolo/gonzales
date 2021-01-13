import Foundation

struct GaussianFilter: Filter {

        init(withSupport support: Vector2F, withAlpha alpha: FloatX) {
                self.support = support
                self.alpha = alpha
                exponent.0 = exp(-alpha * support.x * support.x)
                exponent.1 = exp(-alpha * support.y * support.y)
        }

        func gaussian(x: FloatX, secondTerm: FloatX) -> FloatX {
                return max(0, exp(-alpha * x * x) - secondTerm)
        }

        func evaluate(atLocation point: Point2F) -> FloatX {
               return gaussian(x: point.x, secondTerm: exponent.0) * gaussian(x: point.y, secondTerm: exponent.1)
        }

        let alpha: FloatX
        let exponent: (FloatX, FloatX)
        var support: Vector2F
}
