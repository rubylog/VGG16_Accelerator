// go_next_stage (act를 다 뿌려준 시점) : 을 활용해 보기

module Accum_controller(
    input clk,
    input rst_n,

    input [1:0] conv_or_fc,
    input [5:0] ch, // from TOP controller
    input [12:0] nk, // from TOP controller, for bias mem control
    input [5:0] max_ch, // from cmem

    input accum_activate,
    //input go_next_stage,

    //output go_next_stage,
    output relu_activate,
    output reg [7:0] pixel_reg_idx // prn : 0 ~ 195
);

    localparam IDLE = 2'b11,
               ACCUM = 2'b10;
    
    localparam CONV = 2'b00,
               FC = 2'b01;

    reg [1:0] state, next_state;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) state <= IDLE;
        else state <= next_state;
    end

    always @(*) begin
        case(state)
            IDLE : begin
                if(accum_activate) next_state = ACCUM;
                else next_state = IDLE;
            end
            ACCUM : begin
                if(conv_or_fc == CONV && clk_counter == 8'd222) next_state = IDLE; // after last input write to regfile
                else if(conv_or_fc == FC && clk_counter == 8'd1) next_state = IDLE;
                else next_state = ACCUM;
            end
            default : next_state = next_state;
        endcase
    end

    reg [7:0] clk_counter;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) clk_counter <= 8'b0;
        else if(state == ACCUM) clk_counter <= clk_counter + 8'b1;
        else clk_counter <= 8'b0; 
    end

    //assign go_next_stage = ((conv_or_fc == CONV && clk_counter == 8'd212) || (conv_or_fc == FC && state == ACCUM));

    assign relu_activate = (conv_or_fc == CONV) ? (ch == max_ch) && (clk_counter + 1) % 32 == 0 : (ch == max_ch) && (clk_counter == 2);
    // take two rows during (1clk) relu_active is high

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) pixel_reg_idx <= 8'b0;
        else if(state == ACCUM && conv_or_fc == CONV) begin
            else if((clk_counter + 2) % 16 == 0)  pixel_reg_idx <= pixel_reg_idx;
            else if((clk_counter + 1) % 16 == 0)  pixel_reg_idx <= pixel_reg_idx;
            else if(clk_counter % 16 == 0)  pixel_reg_idx <= pixel_reg_idx;
            else pixel_reg_idx <= pixel_reg_idx + 8'b1;
        end
        else pixel_reg_idx <= 8'b0;
    end

endmodule