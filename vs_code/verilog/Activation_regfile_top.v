module Activation_regfile_top#(
    parameter DATA_WIDTH = 16
)(
    input clk,
    input rst_n,
    input act_load,
    input [DATA_WIDTH*64-1:0] data_first_row, // 64 "channels"
    input [DATA_WIDTH*64-1:0] data_second_row,
    input [DATA_WIDTH*64-1:0] data_third_row,

    output [DATA_WIDTH*64*9-1:0] sliding_patch
);

    genvar i;
    generate
        for(i = 0; i < 64; i = i + 1) begin : Act_regfile_ch
            Activation_regfile_x9 #(DATA_WIDTH) act_regfile_x9(
                clk, rst_n, act_load, 
                data_first_row[DATA_WIDTH*64 - 1 - DATA_WIDTH*i -: DATA_WIDTH],
                data_second_row[DATA_WIDTH*64 - 1 - DATA_WIDTH*i -: DATA_WIDTH],
                data_third_row[DATA_WIDTH*64 - 1 - DATA_WIDTH*i -: DATA_WIDTH],
                sliding_patch[DATA_WIDTH*64*9 - 1 - DATA_WIDTH*9*i -: DATA_WIDTH*9]); // sliding_patch[DATA_WIDTH*64*9 -: DATA_WIDTH*9] is first channel of activation (for 3 x 3 patch)
        end
    endgenerate

endmodule