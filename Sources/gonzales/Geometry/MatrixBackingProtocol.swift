/**
        A type that provides the backing for a matrix, i.e. the type
        it is based on like Float or Double. Float32 or Float16 matrices
        are less precise but also more space efficient.
*/
public protocol MatrixBackingProtocol {

        associatedtype BackingFloat: BinaryFloatingPoint

        init()
        subscript(row: Int, column: Int) -> BackingFloat { get set }
}

