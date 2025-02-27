module PE_top#(
    parameter DATA_WIDTH = 16
)(
    input clk, rst_n,
    input [2:0] ker_load,
    input [DATA_WIDTH*3*64-1:0] weight_from_mem,
    input [DATA_WIDTH*9*64-1:0] activation,
    output [DATA_WIDTH*9*64-1:0] multiply
);

    genvar i;
    generate
        for(i = 0; i < 64; i = i + 1) begin : PE_ch
            PE_x9 #(DATA_WIDTH) pe_x9(
                clk, rst_n, ker_load, 
                weight_from_mem[DATA_WIDTH*3*64 - 1 - DATA_WIDTH*3*i -: DATA_WIDTH*3],
                activation[DATA_WIDTH*9*64 - 1 - DATA_WIDTH*9*i -: DATA_WIDTH*9],
                multiply[DATA_WIDTH*9*64 - 1 - DATA_WIDTH*9*i -: DATA_WIDTH*9]
            )
        end
    endgenerate

endmodule