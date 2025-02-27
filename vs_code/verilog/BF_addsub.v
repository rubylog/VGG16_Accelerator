module BF_adder#(parameter bias = 127)(
    input [15:0] num1, // suppose input is already float type binary
    input [15:0] num2,

    output [15:0] sum
);

// using exp : -14~15
// 00000 : ZERO(-15) / 11111(16) : INF, NaN

localparam ZERO = 15'b000000000000000; // for +0, -0
localparam INF = 15'b111111110000000;

// zero, nan process at input level
wire [1:0] zero;
assign zero[1] = (num1[14:0] == ZERO);
assign zero[0] = (num2[14:0] == ZERO);

wire nan;
assign nan = (num1[14:0] >= INF) || (num2[14:0] >= INF); // output : sign + nan
// Ignoring the intermediate step's result and go to output directly

wire sign1, sign2;
wire [7:0] exp1, exp2; // 8 bits
wire [7:0] frac1, frac2; // 7bits + 1bit (include hidden bit)

// take inputs 
assign sign1 = num1[15];
assign sign2 = num2[15];

// consider non-normalized small value too (not include zero)
assign exp1 = (!num1[14:7] && !zero[1]) ? 8'b1 : num1[14:7];
assign exp2 = (!num2[14:7] && !zero[0]) ? 8'b1 : num2[14:7];
assign frac1 = !num1[14:7] ? {1'b0, num1[6:0]} : {1'b1, num1[6:0]}; // 0.xxxxxx vs 1.xxxxxxxx
assign frac2 = !num2[14:7] ? {1'b0, num2[6:0]} : {1'b1, num2[6:0]};


////////////////////////////// Compare numbers //////////////////////////////

wire sign_big, sign_small;
wire [7:0] exp_big, exp_small;
wire [7:0] frac_big, frac_small; // include hidden bit

localparam NUM1 = 1'b1, NUM2 = 1'b0;
wire find_big;

// Compare absolute value
assign find_big = (exp1 > exp2) ? NUM1 : ((exp1 < exp2) ? NUM2 : ((frac1 > frac2) ? NUM1 : NUM2));
// if abs is same, take num2 as bigger

assign sign_big = find_big ? sign1 : sign2;
assign exp_big = find_big ? exp1 : exp2;
assign frac_big = find_big ? frac1 : frac2;

assign sign_small = find_big ? sign2 : sign1;
assign exp_small = find_big ? exp2 : exp1;
assign frac_small = find_big ? frac2 : frac1;

////////////////////////////// Calculation //////////////////////////////

wire [7:0] exp_diff;
assign exp_diff = exp_big - exp_small;

wire sign_sum;
wire [8:0] exp_sum;
wire [8:0] frac_sum; // 1 bit bigger then inputs, because of carry occur

assign sign_sum = (!frac_sum) ? 1'b0 : sign_big;
// If answer is 0, sign will be fixed as (+)
assign exp_sum = (!frac_sum) ? 9'b000000000 : exp_big;

assign frac_sum = (sign_big == sign_small)
                ? frac_big + (frac_small >> exp_diff)
                : frac_big - (frac_small >> exp_diff);
                // shifted right side bit goes to trash (make error here)

////////////////////////////// Normalization //////////////////////////////

reg [6:0] frac_n; // not include hidden bit

always @(*) begin
    if(!nan) begin
        casex(frac_sum)
           // b1x.xxxxxxx
            9'b1xxxxxxxx: frac_n = frac_sum[7:1];
            9'b01xxxxxxx: frac_n = frac_sum[6:0];
            9'b001xxxxxx: frac_n = {frac_sum[5:0], 1'b0};
            9'b0001xxxxx: frac_n = {frac_sum[4:0], 2'b0};
            9'b00001xxxx: frac_n = {frac_sum[3:0], 3'b0};
            9'b000001xxx: frac_n = {frac_sum[2:0], 4'b0};
            9'b0000001xx: frac_n = {frac_sum[1:0], 5'b0};
            9'b00000001x: frac_n = {frac_sum[0], 6'b0};
            default: frac_n = 7'b0; // 7'b1 or 7'b0
        endcase
    end
end

reg signed [8:0] exp_control; // add to exp_sum

always @(*) begin
    if(!nan) begin
        casex(frac_sum)
           // b1x.xxxxxxx
            9'b1xxxxxxxx: exp_control = 1;
            9'b01xxxxxxx: exp_control = 0; 
            9'b001xxxxxx: exp_control = -1;
            9'b0001xxxxx: exp_control = -2;
            9'b00001xxxx: exp_control = -3;
            9'b000001xxx: exp_control = -4;
            9'b0000001xx: exp_control = -5;
            9'b00000001x: exp_control = -6;
            default: exp_control = 9'b0;
        endcase
    end
end

/////////////////////////////// Exception handling ///////////////////////////////

wire [8:0] exp_n; // add need 1 more bit cuz of carry
assign exp_n = exp_sum + exp_control;

wire underflow;
assign underflow = (exp_control < 0) && (exp_sum < -exp_control);
// only consider underflow by re-normalization

wire overflow;
assign overflow = (exp_n >= 255) && !underflow; // exp_n is unsigned so can find overflow compare with 32

// for overflow and underflow

wire [8:0] err_exp;
assign err_exp = overflow ? 9'b111111111 : (underflow ? 9'b000000000 : exp_n);

wire [7:0] re_nomalized_frac; // size 11 bits to sign expansion fixed to 0
assign re_nomalized_frac = underflow ? {2'b01, frac_n} >> -exp_n : 0; // ex) 1.0001110000 -> 01.0001110000

wire [6:0] err_frac;
assign err_frac = overflow ? 7'b0000000 : (underflow ? re_nomalized_frac[6:0] : frac_n);

assign sum = nan ? {sign_sum, INF} : {sign_sum, err_exp[7:0], err_frac};

endmodule