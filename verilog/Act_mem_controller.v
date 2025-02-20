// addr 접근 패턴 관리가 빡셈
// row num = 1 ~ 14 에 따라서 mux로 mem 1, 2, 3데이터를 reg file의 1,2,3rd row에 연결결

`define CONV 2'b01
`define FC 2'b10

module Act_mem_controller(
    input clk,
    input rst_n,
    input act_load,
    input [1:0] conv_or_fc,
    //input tile,
    input ker_change,

    output reg [12:0] act_mem_addr_1,
    output reg [12:0] act_mem_addr_2,
    output reg [12:0] act_mem_addr_3,
    output en_act_mem,
    output go_next_stage
);

    localparam IDLE = 2'b00,
               MEM_READ_CONV = 2'b01,
               MEM_READ_FC = 2'b10,
               DRAM_ACCESS = 2'b11;


    reg [1:0] state, next_state;

    always @(*) begin
        case(state)
            IDLE : begin
                if(act_load) begin
                    if(conv_or_fc == CONV) next_state = MEM_READ_CONV;
                    else if(conv_or_fc == FC) next_state = MEM_READ_FC;
                    else next_state = IDLE;
                end
                else next_state = IDLE;
            end
            MEM_READ_CONV : begin
                if(sliding_window_counter == 10'd255) next_state = IDLE;
                else next_state = MEM_READ_CONV;
            end
            MEM_READ_FC : begin
                if(fc_counter == 3'd2) next_state = IDLE;
                else next_state = MEM_READ_FC;
            end
            DRAM_ACCESS : begin
                next_state = IDLE;
            end
            default : next_state = next_state;
        endcase
    end

    reg [9:0] sliding_window_counter;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) sliding_window_counter <= 10'd0;
        else if(state == MEM_READ_CONV) sliding_window_counter <= sliding_window_counter + 10'd1;
        else sliding_window_counter <= 10'd0;
    end

    reg [2:0] fc_counter;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) fc_counter <= 3'd0;
        else if(state == MEM_READ_FC) fc_counter <= fc_counter + 3'd1;
        else fc_counter <= 3'd0;
    end

    assign go_next_stage = (conv_or_fc == CONV && sliding_window_counter == 10'd223) || (conv_or_fc == FC && fc_counter == 3'd2);


    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) act_mem_addr_1 <= 13'd0;
        else if(state == MEM_READ_CONV) begin
            //////////////////////////////////// 1st mem 1 line ////////////////////////////////////

            if(sliding_window_counter < 13'd15) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 1

            //////////////////////////////////// 2nd mem 1 line ////////////////////////////////////

            else if(sliding_window_counter < 13'd31) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 2, change

            else if(sliding_window_counter == 13'd31) act_mem_addr_1 <= 13'd16; 
            else if(act_mem_addr_1 < 13'd47) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 3, reuse

            else if(sliding_window_counter == 13'd47) act_mem_addr_1 <= 13'd16;
            else if(sliding_window_counter < 13'd63) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 4, reuse

            //////////////////////////////////// 3rd mem 1 line ////////////////////////////////////

            else if(sliding_window_counter < 13'd79) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 5, change

            else if(sliding_window_counter == 13'd79) act_mem_addr_1 <= 13'd32;
            else if(sliding_window_counter < 13'd95) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 6, reuse

            else if(sliding_window_counter == 13'd95) act_mem_addr_1 <= 13'd32;
            else if(sliding_window_counter < 13'd111) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 7, reuse

            //////////////////////////////////// 4th mem 1 line ////////////////////////////////////

            else if(sliding_window_counter < 13'd127) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 8, change

            else if(sliding_window_counter == 13'd127) act_mem_addr_1 <= 12'd48;
            else if(sliding_window_counter < 13'd143) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 9, reuse

            else if(sliding_window_counter == 13'd143) act_mem_addr_1 <= 12'd48;
            else if(sliding_window_counter < 13'd159) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 10, reuse

            //////////////////////////////////// 5th mem 1 line ////////////////////////////////////

            else if(sliding_window_counter < 13'd175) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 11, change

            else if(sliding_window_counter == 13'd175) act_mem_addr_1 <= 12'd64;
            else if(sliding_window_counter < 13'd191) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 12, reuse

            else if(sliding_window_counter == 13'd191) act_mem_addr_1 <= 12'd64;
            else if(sliding_window_counter < 13'd207) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 13, reuse

            //////////////////////////////////// 5th mem 1 line ////////////////////////////////////

            else if(sliding_window_counter < 13'd175) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 11, change

            else if(sliding_window_counter == 13'd175) act_mem_addr_1 <= 12'd64;
            else if(sliding_window_counter < 13'd191) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 12, reuse

            else if(sliding_window_counter == 13'd191) act_mem_addr_1 <= 12'd64;
            else if(sliding_window_counter < 13'd207) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 13, reuse

            //////////////////////////////////// 5th mem 1 line ////////////////////////////////////

            else if(sliding_window_counter < 13'd223) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 14, change

            /*
            else if(sliding_window_counter == 13'd223) act_mem_addr_1 <= 12'd80;
            else if(sliding_window_counter < 13'd239) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 15, reuse

            else if(sliding_window_counter == 13'd239) act_mem_addr_1 <= 12'd80;
            else if(sliding_window_counter < 13'd255) act_mem_addr_1 <= act_mem_addr_1 + 13'd1; // 16, reuse
            */

            else act_mem_addr_1 <= act_mem_addr_1;
        end
        else if(state == FC) begin
            if(fc_counter < 3'd3) act_mem_addr_1 <= act_mem_addr_1 + 13'd1;
            else act_mem_addr_1 <= act_mem_addr_1;
        end
        else if(state == DRAM_ACCESS) act_mem_addr_1 <= 13'd0;
        else if(state == IDLE) begin
            if(ker_change) act_mem_addr_1 <= 13'd0;
            else act_mem_addr_1 <= act_mem_addr_1;
        end
        else act_mem_addr_1 <= act_mem_addr_1;
    end

endmodule