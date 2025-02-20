// NO POOLING PACKING : row가 7개라서 4개, 3개씩 패킹하면 숫자가 안 맞음

module Out_mem_controller(
    input clk,
    input rst_n,

    input [1:0] conv_or_fc,
    //input pooling,
    input relu_activate,
    input DRAM_access_required,

    output [12:0] out_mem_addr,
    output en_out_mem,
    output wen_out_mem,
    output out_buf_ready
);

    localparam IDLE = 3'b010,
               MEM_WRITE = 3'b011,
               DRAM_ACCESS = 3'b100;

    localparam CONV = 2'b00,
               FC = 2'b01;

    reg [2:0] state, next_state;
    reg [3:0] clk_counter_idle;
    reg [3:0] clk_counter_mem_write;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) clk_counter_idle <= 4'b0;
        else if(state == IDLE && relu_activate) clk_counter_idle <= clk_counter_idle + 4'b1;
        else clk_counter_idle <= 4'b0;
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) clk_counter_mem_write <= 4'b0;
        else if(state == MEM_WRITE) clk_counter_mem_write <= clk_counter_mem_write + 4'b1;
        else clk_counter_mem_write <= 4'b0;
    end
    always @(*) begin
        case(state)
            IDLE : begin
                if(relu_activate) begin
                    if(clk_counter_idle == 4'd2) next_state = MEM_WRITE;
                    //else if(!pooling && clk_counter_idle == 4'd5) next_state = MEM_WRITE;
                    else next_state = IDLE;
                end
                else next_state = IDLE;
            end
            MEM_WRITE : begin
                if(DRAM_access_required) begin
                    if(conv_or_fc == CONV && (out_mem_addr + 1) % 7 == 0) next_state = DRAM_ACCESS;
                    else if(conv_or_fc == FC) next_state = DRAM_ACCESS;
                    else next_state = MEM_WRITE;
                end
                else if(conv_or_fc == CONV && (out_mem_addr + 1) % 7 == 0) next_state = IDLE;
                else if(conv_or_fc == FC) next_state = IDLE;
                else next_state = MEM_WRITE;
            end
            DRAM_ACCESS : begin
                next_state = IDLE;
            end
            default : next_state = next_state;
        endcase
    end

    assign out_buf_ready = (state == DRAM_ACCESS); // maintain 1 clk

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) out_mem_addr <= 13'b0;
        else if(state == MEM_WRITE) begin
            //if(!pooling) 
            out_mem_addr <= out_mem_addr + 13'b1;
            //else if(pooling && ((clk_counter_mem_write + 1) % 4) == 0) out_mem_addr <= out_mem_addr + 13'b1; // consider depth=4 pooling buffer
            else out_mem_addr <= out_mem_addr;
        end
        else if(state == DRAM_ACCESS) out_mem_addr <= 13'b0;
        else out_mem_addr <= out_mem_addr;
    end

    assign en_out_mem = (state == MEM_WRITE);
    assign wen_out_mem = (state == MEM_WRITE);


endmodule

