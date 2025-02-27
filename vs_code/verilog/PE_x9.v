module PE_x9#(
    parameter DATA_WIDTH = 16
)(
    input clk, rst_n, 
    input [2:0] ker_load, // 100 010 001 000
    input [DATA_WIDTH*3-1:0] weight_from_mem,
    input [DATA_WIDTH*9-1:0] activation,
    output [DATA_WIDTH*9-1:0] multiply
);

    wire [DATA_WIDTH*3-1:0] weight_line_one;
    wire [DATA_WIDTH*3-1:0] weight_line_two;
    wire [DATA_WIDTH*3-1:0] weight_line_three;
    
    assign weight_line_one = (ker_load == 3'b100) ? weight_from_mem : (DATA_WIDTH*3){1'b0};
    assign weight_line_two = (ker_load == 3'b010) ? weight_from_mem : (DATA_WIDTH*3){1'b0};
    assign weight_line_three = (ker_load == 3'b001) ? weight_from_mem : (DATA_WIDTH*3){1'b0};

    PE_element #(DATA_WIDTH) one_1(clk, rst_n, ker_load[2], weight_line_one[DATA_WIDTH*3-1 -: DATA_WIDTH], activation[DATA_WIDTH*9-1 -: DATA_WIDTH], multiply[DATA_WIDTH*9-1 -: DATA_WIDTH]);
    PE_element #(DATA_WIDTH) one_2(clk, rst_n, ker_load[2], weight_line_one[DATA_WIDTH*2-1 -: DATA_WIDTH], activation[DATA_WIDTH*8-1 -: DATA_WIDTH], multiply[DATA_WIDTH*8-1 -: DATA_WIDTH]);
    PE_element #(DATA_WIDTH) one_3(clk, rst_n, ker_load[2], weight_line_one[DATA_WIDTH*1-1 -: DATA_WIDTH], activation[DATA_WIDTH*7-1 -: DATA_WIDTH], multiply[DATA_WIDTH*7-1 -: DATA_WIDTH]);

    PE_element #(DATA_WIDTH) two_1(clk, rst_n, ker_load[1], weight_line_two[DATA_WIDTH*3-1 -: DATA_WIDTH], activation[DATA_WIDTH*6-1 -: DATA_WIDTH], multiply[DATA_WIDTH*6-1 -: DATA_WIDTH]);
    PE_element #(DATA_WIDTH) two_2(clk, rst_n, ker_load[1], weight_line_two[DATA_WIDTH*2-1 -: DATA_WIDTH], activation[DATA_WIDTH*5-1 -: DATA_WIDTH], multiply[DATA_WIDTH*5-1 -: DATA_WIDTH]);
    PE_element #(DATA_WIDTH) two_3(clk, rst_n, ker_load[1], weight_line_two[DATA_WIDTH*1-1 -: DATA_WIDTH], activation[DATA_WIDTH*4-1 -: DATA_WIDTH], multiply[DATA_WIDTH*4-1 -: DATA_WIDTH]);

    PE_element #(DATA_WIDTH) three_1(clk, rst_n, ker_load[0], weight_line_three[DATA_WIDTH*3-1 -: DATA_WIDTH], activation[DATA_WIDTH*3-1 -: DATA_WIDTH], multiply[DATA_WIDTH*3-1 -: DATA_WIDTH]);
    PE_element #(DATA_WIDTH) three_2(clk, rst_n, ker_load[0], weight_line_three[DATA_WIDTH*2-1 -: DATA_WIDTH], activation[DATA_WIDTH*2-1 -: DATA_WIDTH], multiply[DATA_WIDTH*2-1 -: DATA_WIDTH]);
    PE_element #(DATA_WIDTH) three_3(clk, rst_n, ker_load[0], weight_line_three[DATA_WIDTH*1-1 -: DATA_WIDTH], activation[DATA_WIDTH*1-1 -: DATA_WIDTH], multiply[DATA_WIDTH*1-1 -: DATA_WIDTH]);

endmodule