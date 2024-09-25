import Foundation

// Adapted from https://github.com/kodecocodes/swift-algorithm-club/blob/master/Binary%20Search/BinarySearch.swift
public func lowerBound<T: Comparable>(_ a: [T], key: T) -> (Int, T) {
        var lowerBound = 0
        var upperBound = a.count
        while lowerBound < upperBound {
                let midIndex = lowerBound + (upperBound - lowerBound) / 2
                if a[midIndex] == key {
                        return (midIndex, a[midIndex])
                } else if a[midIndex] < key {
                        lowerBound = midIndex + 1
                } else {
                        upperBound = midIndex
                }
        }
        return (lowerBound, a[lowerBound])
}

final class PowerLightSampler {

        @MainActor
        init(sampler: Sampler, lights: [Light]) {
                self.sampler = sampler
                self.lights = lights
                totalPower = lights.reduce(0, { total, light in total + light.power() })
                for (i, light) in lights.enumerated() {
                        if i == 0 {
                                cumulativePowers.append(light.power())
                        } else {
                                cumulativePowers.append(cumulativePowers.last! + light.power())
                        }
                }
        }

        @MainActor
        func chooseLight() -> (Light, FloatX) {
                assert(lights.count > 0)
                let u = sampler.get1D()
                let powerIndex = u * totalPower
                let (i, _) = lowerBound(cumulativePowers, key: powerIndex)
                let light = lights[i]
                let probabilityDensity = light.power() / totalPower
                return (light, probabilityDensity)
        }

        let sampler: Sampler
        let lights: [Light]
        var cumulativePowers: [FloatX] = []
        let totalPower: FloatX
}
