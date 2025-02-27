module PE_element#(
    parameter DATA_WIDTH = 16
)(
    input clk, rst_n, ker_load,
    input [DATA_WIDTH-1:0] weight,
    input [DATA_WIDTH-1:0] activation,
    output reg [DATA_WIDTH-1:0] multiply
);

    reg [DATA_WIDTH-1:0] weight_reg;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) weight_reg <= 16'b0;
        else if (ker_load == 1'b1) weight_reg <= weight; // load new data from w mem
        else weight_reg <= weight_reg;
    end

    BF_multiplier multiply(weight_reg, activation, multiply);

endmodule