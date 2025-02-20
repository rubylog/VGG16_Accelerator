// w, act addr 관리 추가
// act regfile controller 만들기
//

module Top_controller(
    input clk,
    input rst_n,

    ///////////////// INPUT ports /////////////////

    // signal from cmem

    input [5:0] max_ch, // max conv ch 512/64 = 8, max fc ch 25088/(9*64) = 44 
    input [12:0] max_ker,
    input [8:0] max_tile,

    input max_ker_on_bram,
    input max_tile_on_bram,

    input start_layer,
    input finish_DRAM_access,

    // signal from accumulator controller
    input go_next_stage,

    // signal from OUT mem controller
    input out_buf_ready,

    ///////////////// OUTPUT ports /////////////////

    // signal to OUT mem controller
    output DRAM_access_required,

    // signal to Accumulator controller
    output accum_activate,

    // signal to cmem
    output dram_access_start,
    output layer_finished,

    // signal to PE
    output reg [2:0] ker_load,

    // signal to bias mem
    output [12:0] bias_mem_addr, // FC max : 0 ~ 4095
    output en_bias_mem,

    // signal to weight mem
    output reg [12:0] weigth_mem_addr,
    output en_weight_mem,

    // signal to activation mem controller
    output act_load,
    output reg [8:0] tile, // 0 ~ 16**2 = 256
    output ker_change
);

    /////////////////////////////////// State declaration ///////////////////////////////////

    localparam IDLE = 3'b000,
               DRAM_ACCESS = 3'b001,
               LOAD_TO_PE = 3'b010,
               TRANSFER = 3'b011,
               ACCUM = 3'b100,
               ONE_KER_END = 3'b101,
               ONE_TILE_END = 3'b110;

    reg [2:0] state, next_state;

    /////////////////////////////////// TOP FSM ///////////////////////////////////

    always @(*) begin
        case(state)
            IDLE : begin
                if(start_layer) next_state = DRAM_ACCESS;
                else next_state = IDLE;
            end
            DRAM_ACCESS : begin
                if(finish_DRAM_access) next_state = LOAD_TO_PE;
                else next_state = DRAM_ACCESS;
            end
            LOAD_TO_PE : begin
                if(counter == 4'd3) next_state = ACCUM; // + include state change 1clk & mem first read 1clk : clk = start of 0 ~ end of 3
                else next_state = LOAD_TO_PE;
            end
            TRANSFER : begin
                if(counter == 4'd13) next_state = ACCUM; // + include state change 1clk : clk = start of 4 ~ end of 13
                else next_state = TRANSFER;
            end
            ACCUM : begin
                if(ch < max_ch && go_next_stage) next_state = LOAD_TO_PE;
                else if(ch == max_ch && go_next_stage) next_state = ONE_KER_END;
                else next_state = ACCUM;
            end
            ONE_KER_END : begin
                if(nk < max_ker && DRAM_access_required && out_buf_ready) next_state = DRAM_ACCESS;
                else if(nk < max_ker && !DRAM_access_required) next_state = LOAD_TO_PE;
                else if(nk == max_ker) next_state = ONE_TILE_END;
                else next_state = ONE_KER_END;
            end
            ONE_TILE_END : begin
                if(tile < max_tile && DRAM_access_required && out_buf_ready) next_state = DRAM_ACCESS;
                else if(tile < max_tile && !DRAM_access_required) next_state = LOAD_TO_PE;
                else if(tile == max_tile) next_state = IDLE;
                else next_state = ONE_TILE_END;
            end
            default : next_state = next_state;
        endcase
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) state <= IDLE;
        else state <= next_state;
    end

    /////////////////////////////////// Internal parameter control ///////////////////////////////////

    reg [3:0] counter; // for checking first data reach to the accumulator

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) counter <= 4'b0;
        else if(state == LOAD_TO_PE || state == TRANSFER) counter <= counter + 4'b1;
        else counter <= 4'b0;
    end

    reg [5:0] ch; // 0 ~ 44
    reg [12:0] nk; // 0 ~ 4096

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) ch <= 6'b0;
        else if(state == LOAD_TO_PE) ch <= ch + 6'b1;
        else if(state == ONE_KER_END) ch <= 6'b0;
        else ch <= ch;
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) nk <= 13'b0;
        else if(state == ONE_KER_END) nk <= nk + 13'b1;
        else if(state = ONE_TILE_END) nk <= 13'b0;
        else nk <= nk;
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) tile <= 9'b1;
        else if(state == ONE_TILE_END) tile <= tile + 9'b1;
        else if(state == IDLE && start_layer) tile <= 9'b1;
        else tile <= tile;
    end

    /////////////////////////////////// OUTPUT signal assingment ///////////////////////////////////

    assign DRAM_access_required = ~((nk % max_ker_on_bram) && (tile % max_tile_on_bram));
    assign accum_activate = (state == ACCUM); 
    assign dram_access_start = (state == DRAM_ACCESS);
    assign layer_finished = (tile == max_tile);

    always @(posedge clk, negedge rst_n) begin // Since it takes 1 clock to read memory anyway, ker_load set it to reg for timing matching.
        if(!rst_n) ker_load <= 3'b000;
        else if(state == LOAD_TO_PE) begin
            case(counter)
                4'd0 : ker_load <= 3'b100;
                4'd1 : ker_load <= 3'b010;
                4'd2 : ker_load <= 3'b001;
                default : ker_load <= 3'b000;
            endcase
        end
        else ker_load = 3'b000;
    end

    assign bias_mem_addr = nk;
    assign en_bias_mem = (ch == 6'b1); // read bias mem before ACCUM

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) weigth_mem_addr <= 13'b0;
        else if(state == LOAD_TO_PE) weigth_mem_addr <= weigth_mem_addr + 13'b1;
        else if(state == ONE_TILE_END || state == DRAM_ACCESS) weigth_mem_addr <= 13'b0;
        else weigth_mem_addr <= weigth_mem_addr;
    end

    assign en_weight_mem = (state == LOAD_TO_PE);

    assign ker_change = (state == ONE_KER_END);

endmodule