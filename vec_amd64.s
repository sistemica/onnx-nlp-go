#include "textflag.h"

// Float32 constants as IEEE-754 bit patterns, in the order of the vecConst names below.
DATA vecConst<>+0(SB)/4, $0xc2aeac4a // lo = -87.3365
DATA vecConst<>+4(SB)/4, $0x42b0c0a5 // hi = 88.3762626647949
DATA vecConst<>+8(SB)/4, $0x3fb8aa3b // log2e = 1.4426950408889634
DATA vecConst<>+12(SB)/4, $0x3f000000 // half = 0.5
DATA vecConst<>+16(SB)/4, $0x3f800000 // one = 1.0
DATA vecConst<>+20(SB)/4, $0x3f318000 // c1 = 0.693359375
DATA vecConst<>+24(SB)/4, $0xb95e8083 // c2 = -0.00021219444
DATA vecConst<>+28(SB)/4, $0x39506967 // p0 = 0.000198756915
DATA vecConst<>+32(SB)/4, $0x3ab743ce // p1 = 0.0013981999507
DATA vecConst<>+36(SB)/4, $0x3c088908 // p2 = 0.0083334519073
DATA vecConst<>+40(SB)/4, $0x3d2aa9c1 // p3 = 0.041665795894
DATA vecConst<>+44(SB)/4, $0x3e2aaaaa // p4 = 0.16666665459
DATA vecConst<>+48(SB)/4, $0x3f000000 // p5 = 0.50000001201
DATA vecConst<>+52(SB)/4, $0xc0800000 // erfmin = -4.0
DATA vecConst<>+56(SB)/4, $0x40800000 // erfmax = 4.0
DATA vecConst<>+60(SB)/4, $0xaf95df0b // n0 = -2.72614225801306e-10
DATA vecConst<>+64(SB)/4, $0x32edffee // n1 = 2.77068142495902e-08
DATA vecConst<>+68(SB)/4, $0xb60cff51 // n2 = -2.10102402082508e-06
DATA vecConst<>+72(SB)/4, $0xb86ec2d2 // n3 = -5.69250639462346e-05
DATA vecConst<>+76(SB)/4, $0xba40ac63 // n4 = -0.000734990630326855
DATA vecConst<>+80(SB)/4, $0xbb41a1f6 // n5 = -0.00295459980854025
DATA vecConst<>+84(SB)/4, $0xbc83dbd4 // n6 = -0.0160960333262415
DATA vecConst<>+88(SB)/4, $0xb77460cd // d0 = -1.45660718464996e-05
DATA vecConst<>+92(SB)/4, $0xb95fbd29 // d1 = -0.000213374055278905
DATA vecConst<>+96(SB)/4, $0xbadc924e // d2 = -0.00168282697438203
DATA vecConst<>+100(SB)/4, $0xbbf19bf8 // d3 = -0.00737332916720468
DATA vecConst<>+104(SB)/4, $0xbc69b6a7 // d4 = -0.0142647390514189
DATA vecConst<>+108(SB)/4, $127 // exponent bias
GLOBL vecConst<>(SB), RODATA|NOPTR, $112

// EXP8 sets Y0 = exp(Y0) for eight lanes. Y15=lo Y14=hi Y13=log2e Y12=half Y11=c1 Y10=c2 Y9=one Y8=127; clobbers Y1-Y5.
#define EXP8 \
	VMAXPS Y15, Y0, Y0; \
	VMINPS Y14, Y0, Y0; \
	VMULPS Y13, Y0, Y1; \
	VADDPS Y12, Y1, Y1; \
	VROUNDPS $1, Y1, Y1; \
	VCVTPS2DQ Y1, Y2; \
	VFNMADD231PS Y11, Y1, Y0; \
	VFNMADD231PS Y10, Y1, Y0; \
	VMULPS Y0, Y0, Y3; \
	VBROADCASTSS vecConst<>+28(SB), Y4; \
	VBROADCASTSS vecConst<>+32(SB), Y5; \
	VFMADD213PS Y5, Y0, Y4; \
	VBROADCASTSS vecConst<>+36(SB), Y5; \
	VFMADD213PS Y5, Y0, Y4; \
	VBROADCASTSS vecConst<>+40(SB), Y5; \
	VFMADD213PS Y5, Y0, Y4; \
	VBROADCASTSS vecConst<>+44(SB), Y5; \
	VFMADD213PS Y5, Y0, Y4; \
	VBROADCASTSS vecConst<>+48(SB), Y5; \
	VFMADD213PS Y5, Y0, Y4; \
	VFMADD213PS Y0, Y3, Y4; \
	VADDPS Y9, Y4, Y4; \
	VPADDD Y8, Y2, Y2; \
	VPSLLD $23, Y2, Y2; \
	VMULPS Y2, Y4, Y0

// LOADEXPCONSTS broadcasts the exp constants into Y8-Y15.
#define LOADEXPCONSTS \
	VBROADCASTSS vecConst<>+0(SB), Y15; \
	VBROADCASTSS vecConst<>+4(SB), Y14; \
	VBROADCASTSS vecConst<>+8(SB), Y13; \
	VBROADCASTSS vecConst<>+12(SB), Y12; \
	VBROADCASTSS vecConst<>+20(SB), Y11; \
	VBROADCASTSS vecConst<>+24(SB), Y10; \
	VBROADCASTSS vecConst<>+16(SB), Y9; \
	VPBROADCASTD vecConst<>+108(SB), Y8

// ERF8 sets Y0 = erf(Y0) for eight lanes with the rational approximation of erf32. Y14=-4 Y15=4; clobbers Y1-Y3, Y5.
#define ERF8 \
	VMAXPS Y14, Y0, Y0; \
	VMINPS Y15, Y0, Y0; \
	VMULPS Y0, Y0, Y1; \
	VBROADCASTSS vecConst<>+60(SB), Y2; \
	VBROADCASTSS vecConst<>+64(SB), Y5; \
	VFMADD213PS Y5, Y1, Y2; \
	VBROADCASTSS vecConst<>+68(SB), Y5; \
	VFMADD213PS Y5, Y1, Y2; \
	VBROADCASTSS vecConst<>+72(SB), Y5; \
	VFMADD213PS Y5, Y1, Y2; \
	VBROADCASTSS vecConst<>+76(SB), Y5; \
	VFMADD213PS Y5, Y1, Y2; \
	VBROADCASTSS vecConst<>+80(SB), Y5; \
	VFMADD213PS Y5, Y1, Y2; \
	VBROADCASTSS vecConst<>+84(SB), Y5; \
	VFMADD213PS Y5, Y1, Y2; \
	VBROADCASTSS vecConst<>+88(SB), Y3; \
	VBROADCASTSS vecConst<>+92(SB), Y5; \
	VFMADD213PS Y5, Y1, Y3; \
	VBROADCASTSS vecConst<>+96(SB), Y5; \
	VFMADD213PS Y5, Y1, Y3; \
	VBROADCASTSS vecConst<>+100(SB), Y5; \
	VFMADD213PS Y5, Y1, Y3; \
	VBROADCASTSS vecConst<>+104(SB), Y5; \
	VFMADD213PS Y5, Y1, Y3; \
	VMULPS Y2, Y0, Y2; \
	VDIVPS Y3, Y2, Y0

// func expSumAVX2(dst, src *float32, n int, shift float32) float32
TEXT ·expSumAVX2(SB), NOSPLIT, $0-36
	MOVQ         dst+0(FP), DI
	MOVQ         src+8(FP), SI
	MOVQ         n+16(FP), CX
	VBROADCASTSS shift+24(FP), Y7
	VXORPS       Y6, Y6, Y6
	LOADEXPCONSTS
expsum_loop:
	CMPQ    CX, $8
	JLT     expsum_done
	VMOVUPS (SI), Y0
	VSUBPS  Y7, Y0, Y0
	EXP8
	VMOVUPS Y0, (DI)
	VADDPS  Y0, Y6, Y6
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $8, CX
	JMP     expsum_loop
expsum_done:
	VEXTRACTF128 $1, Y6, X1
	VADDPS       X1, X6, X6
	VHADDPS      X6, X6, X6
	VHADDPS      X6, X6, X6
	MOVSS        X6, ret+32(FP)
	VZEROUPPER
	RET

// func erfAVX2(dst, src *float32, n int)
TEXT ·erfAVX2(SB), NOSPLIT, $0-24
	MOVQ         dst+0(FP), DI
	MOVQ         src+8(FP), SI
	MOVQ         n+16(FP), CX
	VBROADCASTSS vecConst<>+52(SB), Y14
	VBROADCASTSS vecConst<>+56(SB), Y15
erf_loop:
	CMPQ    CX, $8
	JLT     erf_done
	VMOVUPS (SI), Y0
	ERF8
	VMOVUPS Y0, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $8, CX
	JMP     erf_loop
erf_done:
	VZEROUPPER
	RET

// func geluAVX2(dst, src *float32, n int, divisor, offset, scale float32)
TEXT ·geluAVX2(SB), NOSPLIT, $0-36
	MOVQ         dst+0(FP), DI
	MOVQ         src+8(FP), SI
	MOVQ         n+16(FP), CX
	VBROADCASTSS divisor+24(FP), Y6
	VBROADCASTSS offset+28(FP), Y7
	VBROADCASTSS scale+32(FP), Y4
	VBROADCASTSS vecConst<>+52(SB), Y14
	VBROADCASTSS vecConst<>+56(SB), Y15
gelu_loop:
	CMPQ    CX, $8
	JLT     gelu_done
	VMOVUPS (SI), Y8
	VDIVPS  Y6, Y8, Y0
	ERF8
	VADDPS  Y7, Y0, Y0
	VMULPS  Y0, Y8, Y0
	VMULPS  Y4, Y0, Y0
	VMOVUPS Y0, (DI)
	ADDQ    $32, SI
	ADDQ    $32, DI
	SUBQ    $8, CX
	JMP     gelu_loop
gelu_done:
	VZEROUPPER
	RET

// func scaleAVX2(dst *float32, n int, factor float32)
TEXT ·scaleAVX2(SB), NOSPLIT, $0-20
	MOVQ         dst+0(FP), DI
	MOVQ         n+8(FP), CX
	VBROADCASTSS factor+16(FP), Y1
scale_loop:
	CMPQ    CX, $8
	JLT     scale_done
	VMULPS  (DI), Y1, Y0
	VMOVUPS Y0, (DI)
	ADDQ    $32, DI
	SUBQ    $8, CX
	JMP     scale_loop
scale_done:
	VZEROUPPER
	RET

