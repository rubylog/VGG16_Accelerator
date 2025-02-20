module Psum_Accumulator#(
    parameter DATA_WIDTH = 16
    parameter TILE_SIZE = 14*14
)(
    input clk, rst_n, accum_switch, in_channel, // accum_switch should be off during for each ker and act loading (3clks) + consider first & last data destination time
    input [1:0] conv_or_fc,
    input [DATA_WIDTH-1:0] Adder_tree_result,
    input [DATA_WIDTH-1:0] bias, // from bias memory
    output finish_accum, // when finish one kernel for one tile
    output [DATA_WIDTH*28-1:0] two_rows
);

    localparam CONV = 2'b01;
    localparam FC = 2'b10;

    reg [DATA_WIDTH-1:0] Psum [TILE_SIZE-1:0];

    reg [7:0] pixel_reg_idx; // 0 ~ 256, tile size = 14^2 = 196
    reg [5:0] channel_iteration; // 0 ~ 63, max 512/64 = 64

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) pixel_reg_idx <= 8'd0;
        if(conv_or_fc == CONV && accum_switch) begin
            else if(pixel_reg_idx < 195) pixel_reg_idx <= pixel_reg_idx + 8'd1;
            else if(pixel_reg_idx == 195) pixel_reg_idx <= 8'd0;
            else pixel_reg_idx <= pixel_reg_idx;
        end
        else pixel_reg_idx <= pixel_reg_idx;
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) channel_iteration <= 6'd0;
        if(conv_or_fc == CONV && accum_switch) begin
            else if(pixel_reg_idx == 195) channel_iteration <= channel_iteration + 6'd1;
            else if(channel_iteration == in_channel / 64) channel_iteration <= 6'd0;
            else channel_iteration <= channel_iteration;
        end
        else if(conv_or_fc == FC && accum_switch) begin
            else if(channel_iteration < in_channel / 576) channel_iteration <= channel_iteration + 6'd1;
            else if(channel_iteration == in_channel / 576) channel_iteration <= 6'd0;
            else channel_iteration <= channel_iteration;
        end
    end

    assign finish_accum = (conv_or_fc == CONV && accum_switch) ? (channel_iteration == in_channel / 64) : (channel_iteration == in_channel / 576); // activate only one clk

    wire [DATA_WIDTH-1:0] bias_or_before;
    reg [DATA_WIDTH-1:0] before;
    wire [DATA_WIDTH-1:0] accumlated_result;

    always @(posedge clk, negedge rst_n) begin
        if (!rst_n) 
            before <= 16'd0;
        else if (accum_switch) begin
            if(conv_or_fc == CONV) begin
                case (pixel_reg_idx)
                    8'd0: before <= Psum[1];
                    8'd1: before <= Psum[2];
                    8'd2: before <= Psum[3];
                    8'd3: before <= Psum[4];
                    8'd4: before <= Psum[5];
                    8'd5: before <= Psum[6];
                    8'd6: before <= Psum[7];
                    8'd7: before <= Psum[8];
                    8'd8: before <= Psum[9];
                    8'd9: before <= Psum[10];
                    8'd10: before <= Psum[11];
                    8'd11: before <= Psum[12];
                    8'd12: before <= Psum[13];
                    8'd13: before <= Psum[14];
                    8'd14: before <= Psum[15];
                    8'd15: before <= Psum[16];
                    8'd16: before <= Psum[17];
                    8'd17: before <= Psum[18];
                    8'd18: before <= Psum[19];
                    8'd19: before <= Psum[20];
                    8'd20: before <= Psum[21];
                    8'd21: before <= Psum[22];
                    8'd22: before <= Psum[23];
                    8'd23: before <= Psum[24];
                    8'd24: before <= Psum[25];
                    8'd25: before <= Psum[26];
                    8'd26: before <= Psum[27];
                    8'd27: before <= Psum[28];
                    8'd28: before <= Psum[29];
                    8'd29: before <= Psum[30];
                    8'd30: before <= Psum[31];
                    8'd31: before <= Psum[32];
                    8'd32: before <= Psum[33];
                    8'd33: before <= Psum[34];
                    8'd34: before <= Psum[35];
                    8'd35: before <= Psum[36];
                    8'd36: before <= Psum[37];
                    8'd37: before <= Psum[38];
                    8'd38: before <= Psum[39];
                    8'd39: before <= Psum[40];
                    8'd40: before <= Psum[41];
                    8'd41: before <= Psum[42];
                    8'd42: before <= Psum[43];
                    8'd43: before <= Psum[44];
                    8'd44: before <= Psum[45];
                    8'd45: before <= Psum[46];
                    8'd46: before <= Psum[47];
                    8'd47: before <= Psum[48];
                    8'd48: before <= Psum[49];
                    8'd49: before <= Psum[50];
                    8'd50: before <= Psum[51];
                    8'd51: before <= Psum[52];
                    8'd52: before <= Psum[53];
                    8'd53: before <= Psum[54];
                    8'd54: before <= Psum[55];
                    8'd55: before <= Psum[56];
                    8'd56: before <= Psum[57];
                    8'd57: before <= Psum[58];
                    8'd58: before <= Psum[59];
                    8'd59: before <= Psum[60];
                    8'd60: before <= Psum[61];
                    8'd61: before <= Psum[62];
                    8'd62: before <= Psum[63];
                    8'd63: before <= Psum[64];
                    8'd64: before <= Psum[65];
                    8'd65: before <= Psum[66];
                    8'd66: before <= Psum[67];
                    8'd67: before <= Psum[68];
                    8'd68: before <= Psum[69];
                    8'd69: before <= Psum[70];
                    8'd70: before <= Psum[71];
                    8'd71: before <= Psum[72];
                    8'd72: before <= Psum[73];
                    8'd73: before <= Psum[74];
                    8'd74: before <= Psum[75];
                    8'd75: before <= Psum[76];
                    8'd76: before <= Psum[77];
                    8'd77: before <= Psum[78];
                    8'd78: before <= Psum[79];
                    8'd79: before <= Psum[80];
                    8'd80: before <= Psum[81];
                    8'd81: before <= Psum[82];
                    8'd82: before <= Psum[83];
                    8'd83: before <= Psum[84];
                    8'd84: before <= Psum[85];
                    8'd85: before <= Psum[86];
                    8'd86: before <= Psum[87];
                    8'd87: before <= Psum[88];
                    8'd88: before <= Psum[89];
                    8'd89: before <= Psum[90];
                    8'd90: before <= Psum[91];
                    8'd91: before <= Psum[92];
                    8'd92: before <= Psum[93];
                    8'd93: before <= Psum[94];
                    8'd94: before <= Psum[95];
                    8'd95: before <= Psum[96];
                    8'd96: before <= Psum[97];
                    8'd97: before <= Psum[98];
                    8'd98: before <= Psum[99];
                    8'd99: before <= Psum[100];
                    8'd100: before <= Psum[101];
                    8'd101: before <= Psum[102];
                    8'd102: before <= Psum[103];
                    8'd103: before <= Psum[104];
                    8'd104: before <= Psum[105];
                    8'd105: before <= Psum[106];
                    8'd106: before <= Psum[107];
                    8'd107: before <= Psum[108];
                    8'd108: before <= Psum[109];
                    8'd109: before <= Psum[110];
                    8'd110: before <= Psum[111];
                    8'd111: before <= Psum[112];
                    8'd112: before <= Psum[113];
                    8'd113: before <= Psum[114];
                    8'd114: before <= Psum[115];
                    8'd115: before <= Psum[116];
                    8'd116: before <= Psum[117];
                    8'd117: before <= Psum[118];
                    8'd118: before <= Psum[119];
                    8'd119: before <= Psum[120];
                    8'd120: before <= Psum[121];
                    8'd121: before <= Psum[122];
                    8'd122: before <= Psum[123];
                    8'd123: before <= Psum[124];
                    8'd124: before <= Psum[125];
                    8'd125: before <= Psum[126];
                    8'd126: before <= Psum[127];
                    8'd127: before <= Psum[128];
                    8'd128: before <= Psum[129];
                    8'd129: before <= Psum[130];
                    8'd130: before <= Psum[131];
                    8'd131: before <= Psum[132];
                    8'd132: before <= Psum[133];
                    8'd133: before <= Psum[134];
                    8'd134: before <= Psum[135];
                    8'd135: before <= Psum[136];
                    8'd136: before <= Psum[137];
                    8'd137: before <= Psum[138];
                    8'd138: before <= Psum[139];
                    8'd139: before <= Psum[140];
                    8'd140: before <= Psum[141];
                    8'd141: before <= Psum[142];
                    8'd142: before <= Psum[143];
                    8'd143: before <= Psum[144];
                    8'd144: before <= Psum[145];
                    8'd145: before <= Psum[146];
                    8'd146: before <= Psum[147];
                    8'd147: before <= Psum[148];
                    8'd148: before <= Psum[149];
                    8'd149: before <= Psum[150];
                    8'd150: before <= Psum[151];
                    8'd151: before <= Psum[152];
                    8'd152: before <= Psum[153];
                    8'd153: before <= Psum[154];
                    8'd154: before <= Psum[155];
                    8'd155: before <= Psum[156];
                    8'd156: before <= Psum[157];
                    8'd157: before <= Psum[158];
                    8'd158: before <= Psum[159];
                    8'd159: before <= Psum[160];
                    8'd160: before <= Psum[161];
                    8'd161: before <= Psum[162];
                    8'd162: before <= Psum[163];
                    8'd163: before <= Psum[164];
                    8'd164: before <= Psum[165];
                    8'd165: before <= Psum[166];
                    8'd166: before <= Psum[167];
                    8'd167: before <= Psum[168];
                    8'd168: before <= Psum[169];
                    8'd169: before <= Psum[170];
                    8'd170: before <= Psum[171];
                    8'd171: before <= Psum[172];
                    8'd172: before <= Psum[173];
                    8'd173: before <= Psum[174];
                    8'd174: before <= Psum[175];
                    8'd175: before <= Psum[176];
                    8'd176: before <= Psum[177];
                    8'd177: before <= Psum[178];
                    8'd178: before <= Psum[179];
                    8'd179: before <= Psum[180];
                    8'd180: before <= Psum[181];
                    8'd181: before <= Psum[182];
                    8'd182: before <= Psum[183];
                    8'd183: before <= Psum[184];
                    8'd184: before <= Psum[185];
                    8'd185: before <= Psum[186];
                    8'd186: before <= Psum[187];
                    8'd187: before <= Psum[188];
                    8'd188: before <= Psum[189];
                    8'd189: before <= Psum[190];
                    8'd190: before <= Psum[191];
                    8'd191: before <= Psum[192];
                    8'd192: before <= Psum[193];
                    8'd193: before <= Psum[194];
                    8'd194: before <= Psum[195];
                    8'd195: before <= Psum[0];
                    default: before <= before;
                endcase
            end
            else if(conv_or_fc == FC) before <= Psum[0];
            else before <= before;
        end
    end

    assign bias_or_before = (channel_iteration == 0) ? bias : before;

    BF_adder accumlate(Adder_tree_result, bias_or_before, accumlated_result);

    genvar i;
    generate
        for (i = 0; i < 196; i = i + 1) begin : psum_block
            always @(posedge clk, negedge rst_n) begin
                if(!rst_n) Psum[i] <= 16'd0;
                else if(conv_or_fc == CONV && accum_switch) begin
                    if (pixel_reg_idx == i) Psum[i] <= accumlated_result;
                    else Psum[i] <= Psum[i];
                end
            end
        end
    endgenerate

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) Psum[0] <= 16'd0;
        else if(conv_or_fc == FC && accum_switch) Psum[0] <= accumlated_result;
        else Psum[0] <= Psum[0]
    end

    //assign two_rows[15:0] = (channel_iteration == 63 && pixel_reg_idx == 27) ? Psum[7] : 16'b0;
    //assign two_rows[31:16] = (channel_iteration == 63 && pixel_reg_idx == 27) ? Psum[8] : 16'b0;

    generate
        for (k = 0; k < )
        for (i = 0; i < 14; i = i + 1) begin
            assign two_rows[DATA_WIDTH*28-1 - DATA_WIDTH*i -: DATA_WIDTH] = (channel_iteration == 63 && pixel_reg_idx == 27) ? Psum[i] : 16'b0;
        end
    endgenerate

    generate
        for (i = 0; i < 14; i = i + 1) begin
            assign two_rows[DATA_WIDTH*28-1 - DATA_WIDTH*(i+14) -: DATA_WIDTH] = (channel_iteration == 63 && pixel_reg_idx == 27) ? Psum[27-i] : 16'b0;
        end
    endgenerate



endmodule