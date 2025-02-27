// 곱셈기에만 dsp 쓰기

module BF_multiplier#(parameter bias = 127)(
    input [15:0] num1,
    input [15:0] num2,

    output [15:0] mul
);

localparam ZERO = 15'b000000000000000; // for +0, -0
localparam INF = 15'b111111110000000;

// zero, nan process at input level
wire zero;
wire nan;
assign zero = (num1[14:0] == ZERO) || (num2[14:0] == ZERO); // output : sign + zero
assign nan = (num1[14:0] >= INF) || (num2[14:0] >= INF); // output : sign + nan
// Ignoring the intermediate step's result and go to output directly

wire sign1, sign2;
wire [7:0] exp1, exp2; // 8 bits
wire [7:0] frac1, frac2; // 7bits + 1bit (include hidden bit)

// take inputs 
assign sign1 = num1[15];
assign sign2 = num2[15];

// consider non-normalized small value too (not include zero)
assign exp1 = (!num1[14:7] && !zero) ? 8'b1 : num1[14:7];
assign exp2 = (!num2[14:7] && !zero) ? 8'b1 : num2[14:7];
assign frac1 = (!num1[14:7] && !zero) ? {1'b0, num1[6:0]} : {1'b1, num1[6:0]}; // 0.xxxxxx vs 1.xxxxxxxx
assign frac2 = (!num2[14:7] && !zero) ? {1'b0, num2[6:0]} : {1'b1, num2[6:0]};

////////////////////////////// calculation //////////////////////////////

// sign determine
wire sign_o;
assign sign_o = sign1 ^ sign2;

// exponent calculation
wire [8:0] exp_o; // add need 1 more bit cuz of carry
assign exp_o = exp1 + exp2 - bias;

// fraction multi
wire [15:0] frac_o; // 16bit needed for 8bit * 8bit 
assign frac_o = frac1 * frac2;

/////////////////////////////// normalization ///////////////////////////////
reg [6:0] frac_n; // doesn't include hidden bit

always @(*) begin
    if(!zero && !nan) begin
        casex(frac_o)
            // b1x.xxxxxxxxxxxxxxxxxxx
            16'b1xxxxxxxxxxxxxxx: frac_n = frac_o[14:8];
            16'b01xxxxxxxxxxxxxx: frac_n = frac_o[13:7];
            16'b001xxxxxxxxxxxxx: frac_n = frac_o[12:6];
            16'b0001xxxxxxxxxxxx: frac_n = frac_o[11:5];
            16'b00001xxxxxxxxxxx: frac_n = frac_o[10:4];
            16'b000001xxxxxxxxxx: frac_n = frac_o[9:3];
            16'b0000001xxxxxxxxx: frac_n = frac_o[8:2];
            16'b00000001xxxxxxxx: frac_n = frac_o[7:1];
            16'b000000001xxxxxxx: frac_n = frac_o[6:0];
            16'b0000000001xxxxxx: frac_n = {frac_o[5:0], 1'b0};
            16'b00000000001xxxxx: frac_n = {frac_o[4:0], 2'b0};
            16'b000000000001xxxx: frac_n = {frac_o[3:0], 3'b0};
            16'b0000000000001xxx: frac_n = {frac_o[2:0], 4'b0};
            16'b00000000000001xx: frac_n = {frac_o[1:0], 5'b0};
            16'b000000000000001x: frac_n = {frac_o[0], 6'b0};
            default: frac_n = 7'b0; // for 7'b1 or 7'b0  
        endcase
    end
end

reg signed [8:0] exp_control; // add to exp_o

always @(*) begin
    if(!zero && !nan) begin
        casex(frac_o)
            // b1x.xxxxxxxxxxxxxxxxxxx
            16'b1xxxxxxxxxxxxxxx: exp_control = 1;
            16'b01xxxxxxxxxxxxxx: exp_control = 0;
            16'b001xxxxxxxxxxxxx: exp_control = -1; 
            16'b0001xxxxxxxxxxxx: exp_control = -2;
            16'b00001xxxxxxxxxxx: exp_control = -3;
            16'b000001xxxxxxxxxx: exp_control = -4;
            16'b0000001xxxxxxxxx: exp_control = -5;
            16'b00000001xxxxxxxx: exp_control = -6;
            16'b000000001xxxxxxx: exp_control = -7;
            16'b0000000001xxxxxx: exp_control = -8;
            16'b00000000001xxxxx: exp_control = -9;
            16'b000000000001xxxx: exp_control = -10;
            16'b0000000000001xxx: exp_control = -11;
            16'b00000000000001xx: exp_control = -12;
            16'b000000000000001x: exp_control = -13;
            default: exp_control = 9'b0; 
        endcase 
    end
end

/////////////////////////////// Exception handling ///////////////////////////////

wire [8:0] exp_n;
assign exp_n = exp_o + exp_control;

wire underflow;
assign underflow = (exp1 + exp2 < bias) || ((exp_control < 0) && (exp_o < -exp_control));
// same as exp_o < 0 || exp_n < 0

wire overflow;
assign overflow = (exp_n >= 255) && !underflow; // : exp_n >= 0 1111 1111

// for overflow and underflow

wire [8:0] err_exp; // 9 bits because # bit agreement with exp_n
assign err_exp = overflow ? 9'b111111111 : (underflow ? 9'b000000000 : exp_n);

wire [7:0] re_nomalized_frac; // size 8 bits to sign expansion fixed to 0
assign re_nomalized_frac = underflow ? {2'b01, frac_n} >> -exp_n : 0; // ex) 1.0001110000 -> 01.0001110000 and then shift

wire [6:0] err_frac;
assign err_frac = overflow ? 7'b0000000 : (underflow ? re_nomalized_frac[6:0] : frac_n);

/* 
inputs: 
zero, x -> zero
zero, nan(inf) -> inf(nan) : Nan also means unvalid calculation
nan(inf), nan(inf) -> inf
*/

assign mul = zero && !nan ? {sign_o, ZERO} : (nan ? {sign_o, INF} : {sign_o, err_exp[7:0], err_frac});

endmodule