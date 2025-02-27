module Activation_regfile_x9#(
    parameter DATA_WIDTH = 16
)(
    input clk,
    input rst_n,
    input act_load,
    input [DATA_WIDTH-1:0] data_first_row,
    input [DATA_WIDTH-1:0] data_second_row,
    input [DATA_WIDTH-1:0] data_third_row,

    output [DATA_WIDTH*9-1:0] sliding_patch_wire
);

    reg [DATA_WIDTH-1:0] sliding_patch [8:0]; // reg 0 connected to wire 8

    genvar i;
    generate
        for(i = 0; i < 9; i = i + 1) begin
            assign sliding_patch_wire[DATA_WIDTH*9 - 1 - DATA_WIDTH*i -: DATA_WIDTH] = sliding_patch[i];
        end
    endgenerate

    /////////////////////////////////// Right column ///////////////////////////////////

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[2] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[2] <= data_first_row;
        else sliding_patch[2] <= sliding_patch[2];
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[5] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[5] <= data_second_row;
        else sliding_patch[5] <= sliding_patch[5];
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[8] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[8] <= data_third_row;
        else sliding_patch[8] <= sliding_patch[8];
    end

    /////////////////////////////////// Middle column ///////////////////////////////////

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[1] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[1] <= sliding_patch[2];
        else sliding_patch[1] <= sliding_patch[1];
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[4] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[4] <= sliding_patch[5];
        else sliding_patch[4] <= sliding_patch[4];
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[7] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[7] <= sliding_patch[8];
        else sliding_patch[7] <= sliding_patch[7];
    end

    /////////////////////////////////// Left column ///////////////////////////////////

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[0] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[0] <= sliding_patch[1];
        else sliding_patch[0] <= sliding_patch[0];
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[3] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[3] <= sliding_patch[4];
        else sliding_patch[3] <= sliding_patch[3];
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_patch[6] <= {(DATA_WIDTH){16'b0}};
        else if(act_load) sliding_patch[6] <= sliding_patch[7];
        else sliding_patch[6] <= sliding_patch[6];
    end


endmodule