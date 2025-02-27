module Accum_regfile#(
    parameter DATA_WIDTH = 16
    parameter TILE_SIZE = 14*14
)(
    input [5:0] ch,
    input relu_activate,
    input [7:0] pixel_reg_idx,

    input [DATA_WIDTH-1:0] Adder_tree_result,
    input [DATA_WIDTH-1:0] bias,
    output reg [DATA_WIDTH*28-1:0] two_rows
);

    reg [DATA_WIDTH-1:0] Psum [TILE_SIZE-1:0];

    /////////////////////////////////// Accumlator ///////////////////////////////////

    wire [DATA_WIDTH-1:0] which_add;
    reg [DATA_WIDTH-1:0] previous_Psum;
    wire [DATA_WIDTH-1:0] accum_result;

    /*
    always @(posedge clk, negedge rst_n) begin
        if (!rst_n) previous_Psum <= 16'd0;
        else begin
            case (pixel_reg_idx)
                8'd0: previous_Psum <= Psum[1];
                8'd1: previous_Psum <= Psum[2];
                8'd2: previous_Psum <= Psum[3];
                8'd3: previous_Psum <= Psum[4];
                8'd4: previous_Psum <= Psum[5];
                8'd5: previous_Psum <= Psum[6];
                8'd6: previous_Psum <= Psum[7];
                8'd7: previous_Psum <= Psum[8];
                8'd8: previous_Psum <= Psum[9];
                8'd9: previous_Psum <= Psum[10];
                8'd10: previous_Psum <= Psum[11];
                8'd11: previous_Psum <= Psum[12];
                8'd12: previous_Psum <= Psum[13];
                8'd13: previous_Psum <= Psum[14];
                8'd14: previous_Psum <= Psum[15];
                8'd15: previous_Psum <= Psum[16];
                8'd16: previous_Psum <= Psum[17];
                8'd17: previous_Psum <= Psum[18];
                8'd18: previous_Psum <= Psum[19];
                8'd19: previous_Psum <= Psum[20];
                8'd20: previous_Psum <= Psum[21];
                8'd21: previous_Psum <= Psum[22];
                8'd22: previous_Psum <= Psum[23];
                8'd23: previous_Psum <= Psum[24];
                8'd24: previous_Psum <= Psum[25];
                8'd25: previous_Psum <= Psum[26];
                8'd26: previous_Psum <= Psum[27];
                8'd27: previous_Psum <= Psum[28];
                8'd28: previous_Psum <= Psum[29];
                8'd29: previous_Psum <= Psum[30];
                8'd30: previous_Psum <= Psum[31];
                8'd31: previous_Psum <= Psum[32];
                8'd32: previous_Psum <= Psum[33];
                8'd33: previous_Psum <= Psum[34];
                8'd34: previous_Psum <= Psum[35];
                8'd35: previous_Psum <= Psum[36];
                8'd36: previous_Psum <= Psum[37];
                8'd37: previous_Psum <= Psum[38];
                8'd38: previous_Psum <= Psum[39];
                8'd39: previous_Psum <= Psum[40];
                8'd40: previous_Psum <= Psum[41];
                8'd41: previous_Psum <= Psum[42];
                8'd42: previous_Psum <= Psum[43];
                8'd43: previous_Psum <= Psum[44];
                8'd44: previous_Psum <= Psum[45];
                8'd45: previous_Psum <= Psum[46];
                8'd46: previous_Psum <= Psum[47];
                8'd47: previous_Psum <= Psum[48];
                8'd48: previous_Psum <= Psum[49];
                8'd49: previous_Psum <= Psum[50];
                8'd50: previous_Psum <= Psum[51];
                8'd51: previous_Psum <= Psum[52];
                8'd52: previous_Psum <= Psum[53];
                8'd53: previous_Psum <= Psum[54];
                8'd54: previous_Psum <= Psum[55];
                8'd55: previous_Psum <= Psum[56];
                8'd56: previous_Psum <= Psum[57];
                8'd57: previous_Psum <= Psum[58];
                8'd58: previous_Psum <= Psum[59];
                8'd59: previous_Psum <= Psum[60];
                8'd60: previous_Psum <= Psum[61];
                8'd61: previous_Psum <= Psum[62];
                8'd62: previous_Psum <= Psum[63];
                8'd63: previous_Psum <= Psum[64];
                8'd64: previous_Psum <= Psum[65];
                8'd65: previous_Psum <= Psum[66];
                8'd66: previous_Psum <= Psum[67];
                8'd67: previous_Psum <= Psum[68];
                8'd68: previous_Psum <= Psum[69];
                8'd69: previous_Psum <= Psum[70];
                8'd70: previous_Psum <= Psum[71];
                8'd71: previous_Psum <= Psum[72];
                8'd72: previous_Psum <= Psum[73];
                8'd73: previous_Psum <= Psum[74];
                8'd74: previous_Psum <= Psum[75];
                8'd75: previous_Psum <= Psum[76];
                8'd76: previous_Psum <= Psum[77];
                8'd77: previous_Psum <= Psum[78];
                8'd78: previous_Psum <= Psum[79];
                8'd79: previous_Psum <= Psum[80];
                8'd80: previous_Psum <= Psum[81];
                8'd81: previous_Psum <= Psum[82];
                8'd82: previous_Psum <= Psum[83];
                8'd83: previous_Psum <= Psum[84];
                8'd84: previous_Psum <= Psum[85];
                8'd85: previous_Psum <= Psum[86];
                8'd86: previous_Psum <= Psum[87];
                8'd87: previous_Psum <= Psum[88];
                8'd88: previous_Psum <= Psum[89];
                8'd89: previous_Psum <= Psum[90];
                8'd90: previous_Psum <= Psum[91];
                8'd91: previous_Psum <= Psum[92];
                8'd92: previous_Psum <= Psum[93];
                8'd93: previous_Psum <= Psum[94];
                8'd94: previous_Psum <= Psum[95];
                8'd95: previous_Psum <= Psum[96];
                8'd96: previous_Psum <= Psum[97];
                8'd97: previous_Psum <= Psum[98];
                8'd98: previous_Psum <= Psum[99];
                8'd99: previous_Psum <= Psum[100];
                8'd100: previous_Psum <= Psum[101];
                8'd101: previous_Psum <= Psum[102];
                8'd102: previous_Psum <= Psum[103];
                8'd103: previous_Psum <= Psum[104];
                8'd104: previous_Psum <= Psum[105];
                8'd105: previous_Psum <= Psum[106];
                8'd106: previous_Psum <= Psum[107];
                8'd107: previous_Psum <= Psum[108];
                8'd108: previous_Psum <= Psum[109];
                8'd109: previous_Psum <= Psum[110];
                8'd110: previous_Psum <= Psum[111];
                8'd111: previous_Psum <= Psum[112];
                8'd112: previous_Psum <= Psum[113];
                8'd113: previous_Psum <= Psum[114];
                8'd114: previous_Psum <= Psum[115];
                8'd115: previous_Psum <= Psum[116];
                8'd116: previous_Psum <= Psum[117];
                8'd117: previous_Psum <= Psum[118];
                8'd118: previous_Psum <= Psum[119];
                8'd119: previous_Psum <= Psum[120];
                8'd120: previous_Psum <= Psum[121];
                8'd121: previous_Psum <= Psum[122];
                8'd122: previous_Psum <= Psum[123];
                8'd123: previous_Psum <= Psum[124];
                8'd124: previous_Psum <= Psum[125];
                8'd125: previous_Psum <= Psum[126];
                8'd126: previous_Psum <= Psum[127];
                8'd127: previous_Psum <= Psum[128];
                8'd128: previous_Psum <= Psum[129];
                8'd129: previous_Psum <= Psum[130];
                8'd130: previous_Psum <= Psum[131];
                8'd131: previous_Psum <= Psum[132];
                8'd132: previous_Psum <= Psum[133];
                8'd133: previous_Psum <= Psum[134];
                8'd134: previous_Psum <= Psum[135];
                8'd135: previous_Psum <= Psum[136];
                8'd136: previous_Psum <= Psum[137];
                8'd137: previous_Psum <= Psum[138];
                8'd138: previous_Psum <= Psum[139];
                8'd139: previous_Psum <= Psum[140];
                8'd140: previous_Psum <= Psum[141];
                8'd141: previous_Psum <= Psum[142];
                8'd142: previous_Psum <= Psum[143];
                8'd143: previous_Psum <= Psum[144];
                8'd144: previous_Psum <= Psum[145];
                8'd145: previous_Psum <= Psum[146];
                8'd146: previous_Psum <= Psum[147];
                8'd147: previous_Psum <= Psum[148];
                8'd148: previous_Psum <= Psum[149];
                8'd149: previous_Psum <= Psum[150];
                8'd150: previous_Psum <= Psum[151];
                8'd151: previous_Psum <= Psum[152];
                8'd152: previous_Psum <= Psum[153];
                8'd153: previous_Psum <= Psum[154];
                8'd154: previous_Psum <= Psum[155];
                8'd155: previous_Psum <= Psum[156];
                8'd156: previous_Psum <= Psum[157];
                8'd157: previous_Psum <= Psum[158];
                8'd158: previous_Psum <= Psum[159];
                8'd159: previous_Psum <= Psum[160];
                8'd160: previous_Psum <= Psum[161];
                8'd161: previous_Psum <= Psum[162];
                8'd162: previous_Psum <= Psum[163];
                8'd163: previous_Psum <= Psum[164];
                8'd164: previous_Psum <= Psum[165];
                8'd165: previous_Psum <= Psum[166];
                8'd166: previous_Psum <= Psum[167];
                8'd167: previous_Psum <= Psum[168];
                8'd168: previous_Psum <= Psum[169];
                8'd169: previous_Psum <= Psum[170];
                8'd170: previous_Psum <= Psum[171];
                8'd171: previous_Psum <= Psum[172];
                8'd172: previous_Psum <= Psum[173];
                8'd173: previous_Psum <= Psum[174];
                8'd174: previous_Psum <= Psum[175];
                8'd175: previous_Psum <= Psum[176];
                8'd176: previous_Psum <= Psum[177];
                8'd177: previous_Psum <= Psum[178];
                8'd178: previous_Psum <= Psum[179];
                8'd179: previous_Psum <= Psum[180];
                8'd180: previous_Psum <= Psum[181];
                8'd181: previous_Psum <= Psum[182];
                8'd182: previous_Psum <= Psum[183];
                8'd183: previous_Psum <= Psum[184];
                8'd184: previous_Psum <= Psum[185];
                8'd185: previous_Psum <= Psum[186];
                8'd186: previous_Psum <= Psum[187];
                8'd187: previous_Psum <= Psum[188];
                8'd188: previous_Psum <= Psum[189];
                8'd189: previous_Psum <= Psum[190];
                8'd190: previous_Psum <= Psum[191];
                8'd191: previous_Psum <= Psum[192];
                8'd192: previous_Psum <= Psum[193];
                8'd193: previous_Psum <= Psum[194];
                8'd194: previous_Psum <= Psum[195];
                8'd195: previous_Psum <= Psum[0];
                default: previous_Psum <= previous_Psum;
            endcase
        end
    end
    */

    always @(posedge clk, negedge rst_n) begin
        if (!rst_n) previous_Psum <= 16'd0;
        else if (pixel_reg_idx < 8'd195) 
            previous_Psum <= Psum[pixel_reg_idx + 1];
        else if (pixel_reg_idx == 8'd195)
            previous_Psum <= Psum[0];
        else
            previous_Psum <= previous_Psum;
    end
    
    assign which_add = (ch == 6'b1) ? bias : previous_Psum;

    BF_adder accum(Adder_tree_result, which_add, accum_result);

    /////////////////////////////////// Update Regfile ///////////////////////////////////

    genvar i;
    generate
        for (i = 0; i < 196; i = i + 1) begin : psum_block
            always @(posedge clk, negedge rst_n) begin
                if(!rst_n) Psum[i] <= 16'd0;
                else if (pixel_reg_idx == i) Psum[i] <= accum_result;
                else Psum[i] <= Psum[i];
            end
        end
    endgenerate

    /////////////////////////////////// Relu buffering ///////////////////////////////////

    wire [7:0] i;
    assign i = (pixel_reg_idx >= 8'd27) ? (pixel_reg_idx - 8'd27) : 8'd0;
    
    /*
    always @(posedge clk, negedge rst_n) begin
        if (!rst_n) 
            two_rows <= {DATA_WIDTH*28{1'b0}};
        else if (relu_activate) 
            two_rows <= {Psum[i], Psum[i+1], Psum[i+2], Psum[i+3], Psum[i+4], 
                        Psum[i+5], Psum[i+6], Psum[i+7], Psum[i+8], Psum[i+9], 
                        Psum[i+10], Psum[i+11], Psum[i+12], Psum[i+13], Psum[i+14],
                        Psum[i+15], Psum[i+16], Psum[i+17], Psum[i+18], Psum[i+19],
                        Psum[i+20], Psum[i+21], Psum[i+22], Psum[i+23], Psum[i+24],
                        Psum[i+25], Psum[i+26], Psum[i+27]};
        else 
            two_rows <= two_rows;
    end
    */

    genvar idx;
    always @(posedge clk, negedge rst_n) begin
        if (!rst_n) 
            two_rows <= {DATA_WIDTH*28{1'b0}};
        else if (relu_activate) begin
            for (idx = 0; idx < 28; idx = idx + 1) 
                two_rows[(DATA_WIDTH*(28-idx)-1) -: DATA_WIDTH] <= Psum[i + idx];
        end
    end




endmodule