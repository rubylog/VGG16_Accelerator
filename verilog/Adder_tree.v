module Adder_tree#(
    parameter DATA_WIDTH = 16
)(
    input clk, rst_n,
    input [(DATA_WIDTH*576)-1:0] PE_result, // reg already in each PE's output
    output reg [DATA_WIDTH-1:0] Adder_tree_result
);

localparam PE_RESULT_WIDTH = DATA_WIDTH*576;
localparam Layer1_WIDTH = DATA_WIDTH*288;
localparam Layer2_WIDTH = DATA_WIDTH*144;
localparam Layer3_WIDTH = DATA_WIDTH*72;
localparam Layer4_WIDTH = DATA_WIDTH*36;
localparam Layer5_WIDTH = DATA_WIDTH*18;
localparam Layer6_WIDTH = DATA_WIDTH*9; // add one 0 input, make it 10
localparam Layer7_WIDTH = DATA_WIDTH*5; // " , make it 6 
localparam Layer8_WIDTH = DATA_WIDTH*3; // " , make it 4
localparam Layer9_WIDTH = DATA_WIDTH*2;

wire [Layer1_WIDTH-1:0] layer1_out_wire;
wire [Layer2_WIDTH-1:0] layer2_out_wire;
wire [Layer3_WIDTH-1:0] layer3_out_wire;
wire [Layer4_WIDTH-1:0] layer4_out_wire;
wire [Layer5_WIDTH-1:0] layer5_out_wire;
wire [Layer6_WIDTH-1:0] layer6_out_wire;
wire [Layer7_WIDTH-1:0] layer7_out_wire;
wire [Layer8_WIDTH-1:0] layer8_out_wire;
wire [Layer9_WIDTH-1:0] layer9_out_wire;
wire [DATA_WIDTH-1:0] layer10_out_wire;

///////////////////////////////////// Layer1 /////////////////////////////////////

genvar i;
generate
    for (i = 0; i < 576; i = i + 2) begin : BF_adder_Layer1
        BF_adder Layer1 (PE_result[(PE_RESULT_WIDTH - 1 - DATA_WIDTH*i) -: DATA_WIDTH],  
            PE_result[(PE_RESULT_WIDTH - 1 - DATA_WIDTH*(i+1)) -: DATA_WIDTH], 
            layer1_out_wire[(Layer1_WIDTH - 1 - DATA_WIDTH*(i/2)) -: DATA_WIDTH]);
    end
endgenerate

reg [Layer1_WIDTH-1:0] layer1_out_reg; // FF

///////////////////////////////////// Layer2 /////////////////////////////////////

generate
    for (i = 0; i < 288; i = i + 2) begin : BF_adder_Layer2
        BF_adder Layer2 (layer1_out_reg[(Layer1_WIDTH - 1 - DATA_WIDTH*i) -: DATA_WIDTH],  
            layer1_out_reg[(Layer1_WIDTH - 1 - DATA_WIDTH*(i+1)) -: DATA_WIDTH], 
            layer2_out_wire[(Layer2_WIDTH - 1 - DATA_WIDTH*(i/2)) -: DATA_WIDTH]);
    end
endgenerate

reg [Layer2_WIDTH-1:0] layer2_out_reg; // FF

///////////////////////////////////// Layer3 /////////////////////////////////////

generate
    for (i = 0; i < 144; i = i + 2) begin : BF_adder_Layer3
        BF_adder Layer3 (layer2_out_reg[(Layer2_WIDTH - 1 - DATA_WIDTH*i) -: DATA_WIDTH],  
            layer2_out_reg[(Layer2_WIDTH - 1 - DATA_WIDTH*(i+1)) -: DATA_WIDTH], 
            layer3_out_wire[(Layer3_WIDTH - 1 - DATA_WIDTH*(i/2)) -: DATA_WIDTH]);
    end
endgenerate

reg [Layer3_WIDTH-1:0] layer3_out_reg; // FF

///////////////////////////////////// Layer4 /////////////////////////////////////

generate
    for (i = 0; i < 72; i = i + 2) begin : BF_adder_Layer4
        BF_adder Layer4 (layer3_out_reg[(Layer3_WIDTH - 1 - DATA_WIDTH*i) -: DATA_WIDTH],  
            layer3_out_reg[(Layer3_WIDTH - 1 - DATA_WIDTH*(i+1)) -: DATA_WIDTH], 
            layer4_out_wire[(Layer4_WIDTH - 1 - DATA_WIDTH*(i/2)) -: DATA_WIDTH]);
    end
endgenerate

reg [Layer4_WIDTH-1:0] layer4_out_reg; // FF

///////////////////////////////////// Layer5 /////////////////////////////////////

generate
    for (i = 0; i < 36; i = i + 2) begin : BF_adder_Layer5
        BF_adder Layer5 (layer4_out_reg[(Layer4_WIDTH - 1 - DATA_WIDTH*i) -: DATA_WIDTH],  
            layer4_out_reg[(Layer4_WIDTH - 1 - DATA_WIDTH*(i+1)) -: DATA_WIDTH], 
            layer5_out_wire[(Layer5_WIDTH - 1 - DATA_WIDTH*(i/2)) -: DATA_WIDTH]);
    end
endgenerate

reg [Layer5_WIDTH-1:0] layer5_out_reg; // FF

///////////////////////////////////// Layer6 /////////////////////////////////////

generate
    for (i = 0; i < 18; i = i + 2) begin : BF_adder_Layer6
        BF_adder Layer6 (layer5_out_reg[(Layer5_WIDTH - 1 - DATA_WIDTH*i) -: DATA_WIDTH],  
            layer5_out_reg[(Layer5_WIDTH - 1 - DATA_WIDTH*(i+1)) -: DATA_WIDTH], 
            layer6_out_wire[(Layer6_WIDTH - 1 - DATA_WIDTH*(i/2)) -: DATA_WIDTH]);
    end
endgenerate

reg [Layer6_WIDTH-1:0] layer6_out_reg; // FF

///////////////////////////////////// Layer7 /////////////////////////////////////

generate
    for (i = 0; i < 8; i = i + 2) begin : BF_adder_Layer7
        BF_adder Layer7 (layer6_out_reg[(Layer6_WIDTH - 1 - DATA_WIDTH*i) -: DATA_WIDTH],  
            layer6_out_reg[(Layer6_WIDTH - 1 - DATA_WIDTH*(i+1)) -: DATA_WIDTH], 
            layer7_out_wire[(Layer7_WIDTH - 1 - DATA_WIDTH*(i/2)) -: DATA_WIDTH]);
    end
endgenerate

BF_adder Layer7_extra (layer6_out_reg[(Layer6_WIDTH - 1 - DATA_WIDTH*8) -: DATA_WIDTH],  
            16'b0, layer7_out_wire[(Layer7_WIDTH - 1 - DATA_WIDTH*4) -: DATA_WIDTH]);

reg [Layer7_WIDTH-1:0] layer7_out_reg; // FF

///////////////////////////////////// Layer8 /////////////////////////////////////

generate
    for (i = 0; i < 4; i = i + 2) begin : BF_adder_Layer8
        BF_adder Layer8 (layer7_out_reg[(Layer7_WIDTH - 1 - DATA_WIDTH*i) -: DATA_WIDTH],  
            layer7_out_reg[(Layer7_WIDTH - 1 - DATA_WIDTH*(i+1)) -: DATA_WIDTH], 
            layer8_out_wire[(Layer8_WIDTH - 1 - DATA_WIDTH*(i/2)) -: DATA_WIDTH]);
    end
endgenerate

BF_adder Layer8_extra (layer7_out_reg[(Layer7_WIDTH - 1 - DATA_WIDTH*4) -: DATA_WIDTH],  
            16'b0, layer8_out_wire[(Layer8_WIDTH - 1 - DATA_WIDTH*2) -: DATA_WIDTH]);

reg [Layer8_WIDTH-1:0] layer8_out_reg; // FF

///////////////////////////////////// Layer9 /////////////////////////////////////

BF_adder BF_adder_Layer9 (layer8_out_reg[(Layer8_WIDTH - 1) -: DATA_WIDTH], 
                            layer8_out_reg[(Layer8_WIDTH - 1 - DATA_WIDTH) -: DATA_WIDTH], layer9_out_wire[Layer9_WIDTH - 1 -: DATA_WIDTH]);

BF_adder Layer9_extra (layer8_out_reg[(Layer8_WIDTH - 1 - DATA_WIDTH*2) -: DATA_WIDTH], 
                            16'b0, layer9_out_wire[DATA_WIDTH - 1 : 0]);

reg [Layer9_WIDTH-1:0] layer9_out_reg;

///////////////////////////////////// Layer10 /////////////////////////////////////

BF_adder BF_adder_Layer10 (layer9_out_reg[(Layer9_WIDTH - 1) -: DATA_WIDTH],
                            layer9_out_reg[DATA_WIDTH - 1 : 0], layer10_out_wire);
                            

///////////////////////////////////// PIPELINING /////////////////////////////////////

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer1_out_reg <= 0;
    else layer1_out_reg <= layer1_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer2_out_reg <= 0;
    else layer2_out_reg <= layer2_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer3_out_reg <= 0;
    else layer3_out_reg <= layer3_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer4_out_reg <= 0;
    else layer4_out_reg <= layer4_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer5_out_reg <= 0;
    else layer5_out_reg <= layer5_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer6_out_reg <= 0;
    else layer6_out_reg <= layer6_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer7_out_reg <= 0;
    else layer7_out_reg <= layer7_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer8_out_reg <= 0;
    else layer8_out_reg <= layer8_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) layer9_out_reg <= 0;
    else layer9_out_reg <= layer9_out_wire;
end

always @(posedge clk, negedge rst_n) begin
    if(!rst_n) Adder_tree_result <= 0;
    else Adder_tree_result <= layer10_out_wire;
end

endmodule