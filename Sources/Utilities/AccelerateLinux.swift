//
//  AccelerateLinux.swift
//  database-server
//
//  Created by Ritesh Pakala Rao on 2/7/26.
//

import Foundation

typealias vDSP_Length = Int

func vDSP_dotpr(_ x: [Float], _ strideX: Int, _ y: [Float], _ strideY: Int,
                _ result: inout Float, _ length: Int) {
    guard strideX == 1 && strideY == 1 else {
        result = 0.0
        for i in 0..<length { result += x[i * strideX] * y[i * strideY] }
        return
    }
    var s0: Float = 0, s1: Float = 0, s2: Float = 0, s3: Float = 0
    var s4: Float = 0, s5: Float = 0, s6: Float = 0, s7: Float = 0
    var i = 0
    while i &+ 8 <= length {
        s0 += x[i]   * y[i];   s1 += x[i+1] * y[i+1]
        s2 += x[i+2] * y[i+2]; s3 += x[i+3] * y[i+3]
        s4 += x[i+4] * y[i+4]; s5 += x[i+5] * y[i+5]
        s6 += x[i+6] * y[i+6]; s7 += x[i+7] * y[i+7]
        i &+= 8
    }
    while i < length { s0 += x[i] * y[i]; i &+= 1 }
    result = s0+s1+s2+s3+s4+s5+s6+s7
}

func vDSP_dotprD(_ x: [Double], _ strideX: Int, _ y: [Double], _ strideY: Int,
                 _ result: inout Double, _ length: Int) {
    result = 0.0
    for i in 0..<length {
        result += x[i * strideX] * y[i * strideY]
    }
}

func vDSP_vsubD(_ x: [Double], _ strideX: Int, _ y: [Double], _ strideY: Int,
                _ result: inout [Double], _ strideResult: Int, _ length: Int) {
    for i in 0..<length {
        result[i * strideResult] = x[i * strideX] - y[i * strideY]
    }
}
