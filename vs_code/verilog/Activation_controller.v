// 0301 : need to add signal 10, for Layer 5 (all sided zero padding)
// 0302 : need to add MUX (merge between regfile and memory)

module Activation_controller(
    input clk, rst_n,
    input [2:0] conv_layer_num, // 1 ~ 5
    input [7:0] tile_idx, // 0 ~ 255 (Layer 1 : # size 16 tiles = 16 x 16)
    input start_activation_load, ker_change, tile_change, dram_access, // posedge detector

    // signal goes to top control
    output sliding_finish,

    // activation reg-file control
    output act_mem_to_reg, // same as act_load in activation regfile top
    output [1:0] reg_first_row_selection_signal,
    output [1:0] reg_second_row_selection_signal,
    output [1:0] reg_third_row_selection_signal,

    // activation memory control
    output en_A, en_B, en_C,
    output [9:0] addr_A_out, addr_B_out, addr_C_out // depth 1024
);

    //////////////////////////////////////// FSM part ////////////////////////////////////////

    localparam IDLE = 3'b000,
               SLIDING = 3'b001,
               READY_FOR_CHANGE = 3'b010,
               //CHANNEL_CHANGE = 3'b011, // needed?
               KER_CHANGE = 3'b100,
               TILE_CHANGE = 3'b101,
               DRAM_ACCESS = 3'b110;

    reg [2:0] state, next_state;

    always @(*) begin
        case(state)
            IDLE: begin
                if(start_activation_load) next_state = SLIDING;
            end
            SLIDING: begin
                if(sliding_finish) next_state = READY_FOR_CHANGE;
            end
            READY_FOR_CHANGE: begin // 1 clk
                if(ker_change) next_state = KER_CHANGE;
                else if(tile_change) next_state = TILE_CHANGE;
                else if(dram_access) next_state = DRAM_ACCESS;
                else next_state = SLIDING; // only channel change
            end
            KER_CHANGE: begin
                next_state = IDLE;
            end
            TILE_CHANGE: begin
                next_state = IDLE;
            end
            DRAM_ACCESS: begin
                next_state = IDLE;
            end
            default: next_state = state;
        endcase
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) state <= 3'b0;
        else state <= next_state;
    end

    //////////////////////////////////////// zero-padding signal generate ////////////////////////////////////////
    // -> Asynchronous operation (Only LUT based combinational circuit), works even in IDLE state. 
    // : conv_layer_num -> tile_max_coordinates -> tile_xy_coordinates -> zero_padding_signal

    wire [3:0] tile_x_coordinate; // 0 ~ 15
    wire [3:0] tile_y_coordinate; // 0 ~ 15

    reg [3:0] tile_max_coordinate;

    always @(*) begin
        case(conv_layer_num)
            3'd1: tile_max_coordinate = 4'd15;
            3'd2: tile_max_coordinate = 4'd7;
            3'd3: tile_max_coordinate = 4'd3;
            3'd4: tile_max_coordinate = 4'd1;
            3'd5: tile_max_coordinate = 4'd0;
            default: tile_max_coordinate = 4'd0;
        endcase
    end

    // Synthesizable division and modulo operations (LUT method)
    always @(*) begin
        case(tile_max_coordinate)
            4'd15: begin
                tile_x_coordinate = tile_idx & 4'b1111;  // tile_idx % 16
                tile_y_coordinate = tile_idx >> 4;       // tile_idx / 16
            end
            4'd7: begin
                tile_x_coordinate = tile_idx & 4'b0111;  // tile_idx % 8
                tile_y_coordinate = tile_idx >> 3;       // tile_idx / 8
            end
            4'd3: begin
                tile_x_coordinate = tile_idx & 4'b0011;  // tile_idx % 4
                tile_y_coordinate = tile_idx >> 2;       // tile_idx / 4
            end
            4'd1: begin
                tile_x_coordinate = tile_idx & 4'b0001;  // tile_idx % 2
                tile_y_coordinate = tile_idx >> 1;       // tile_idx / 2
            end
            4'd0: begin
                tile_x_coordinate = 4'd0;  // tile_idx % 1 = always 0
                tile_y_coordinate = tile_idx; // tile_idx / 1 = tile_idx
            end
            default: begin
                tile_x_coordinate = 4'd0;
                tile_y_coordinate = 4'd0;
            end
        endcase
    end

    reg [3:0] zero_padding_signal; // wire behave

    always @(*) begin
        if(tile_max_coordinate != 0) begin
            // two sides zero-padding
            if(tile_x_coordinate == 4'd0 && tile_y_coordinate == 4'd0) zero_padding_signal = 4'd1;
            else if(tile_x_coordinate == tile_max_coordinate && tile_y_coordinate == 4'd0) zero_padding_signal = 4'd2;
            else if(tile_x_coordinate == tile_max_coordinate && tile_y_coordinate == tile_max_coordinate) zero_padding_signal = 4'd3;
            else if(tile_x_coordinate == 4'd0 && tile_y_coordinate == tile_max_coordinate) zero_padding_signal = 4'd4;

            // one side zero-padding
            else if(tile_x_coordinate == 4'd0) zero_padding_signal = 4'd5;
            else if(tile_y_coordinate == 4'd0) zero_padding_signal = 4'd5;
            else if(tile_x_coordinate == tile_max_coordinate) zero_padding_signal = 4'd7;
            else if(tile_y_coordinate == tile_max_coordinate) zero_padding_signal = 4'd8;

            else zero_padding_signal = 4'd0; // No padding
        end
        else zero_padding_signal = 4'd9; // case for Layer 5
    end

    //////////////////////////////////////// enable code control ////////////////////////////////////////
    // zero_padding_signal finished -> state == SLIDING -> read_index (+1clk) -> enable_code_pattern (+1clk)

    /*

    enable code mapping

    'ABC' = 0 
    '0AB' = 1
    'CAB' = 2
    'BCA' = 3
    'BC0' = 4
    '000' = 5

    -> 0 or A or B or C which are mapping to 0, 1, 2, 3 in verilog

    */

    localparam ENCODE_BIT_WIDTH = 2, // 2 bits * 3 words
               ZERO = 2'b00,
               A = 2'b01,
               B = 2'b10,
               C = 2'b11;

    reg [ENCODE_BIT_WIDTH*3-1:0] enable_code; // wire behave, five different codes, each has 3 words.
    reg [2:0] encode_idx;

    always @(*) begin
        case (encode_idx)
            3'd0: enable_code = {A, B, C};
            3'd1: enable_code = {ZERO, A, B};
            3'd2: enable_code = {C, A, B};
            3'd3: enable_code = {B, C, A};
            3'd4: enable_code = {B, C, ZERO};
            3'd4: enable_code = {ZERO, ZERO, ZERO};
            default: enable_code = {ZERO, ZERO, ZERO};
        endcase
    end

    localparam ENCODE_IDX_BIT_WIDTH = 3;

    reg [ENCODE_IDX_BIT_WIDTH*14*16-1:0] enable_code_patterns [0:8]; // (14 rows * 16 clks) * 9 signals

    // for the FPGA synthesis, below code be placed by .coe file initialization with "BRAM" IP ROM mode.
    initial begin
        $readmemb("enable_code_patterns.mem", enable_code_patterns); // 9 signals * (14*16)
    end

    // select enable_code_pattern for the generated signal
    reg [ENCODE_IDX_BIT_WIDTH*14*16-1:0] enable_code_pattern_for_signal;

    //************************** The synchronized always block below should only work in sliding state. **************************//

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) enable_code_pattern_for_signal <= {(ENCODE_IDX_BIT_WIDTH*14*16){1'b0}};
        else enable_code_pattern_for_signal <= enable_code_pattern[zero_padding_signal];
    end // bram ROM requires one clk to read. -> this always block represent read only bram.
    //** at least change with state change to SLIDING

    reg [7:0] read_index_1; // 14*16 -> 8 bits required

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) read_index_1 <= ENCODE_IDX_BIT_WIDTH*14*16;
        else if(state == SLIDING) read_index_1 <= read_index_1 - 8'd3;
        else read_index_1 <= ENCODE_IDX_BIT_WIDTH*14*16;
    end
    //** +1clk after SLIDING
    // Need clk cycle condition in SLIDING state to avoid read_index goes to negative number
    
    always @(posedge clk, negedge rst_n) begin
        if (!rst_n) encode_idx <= 3'b000;
        else if(state == SLIDING) encode_idx <= enable_code_pattern_for_signal[read_index_1 -: ENCODE_IDX_BIT_WIDTH]; // read 3 bits
        else encode_idx <= 3'b000;
    end
    //** +1clk after read_index

    // output enable signal assignment
    assign reg_first_row_selection_signal = enable_code[5:4];
    assign reg_second_row_selection_signal = enable_code[3:2];
    assign reg_third_row_selection_signal = enable_code[1:0];

    
    //////////////////////////////////////// addr pattern control ////////////////////////////////////////

    //////////////////////////////
    ///      FOR MEMORY A      ///
    //////////////////////////////

    localparam ADDR_BIT_WIDTH = 7;

    reg [ADDR_BIT_WIDTH*16*14-1:0] A_addr_pattern [8:0]; // max 16 clk * 14 rows * 9 signals

    // for the FPGA synthesis, below code be placed by .coe file initialization with "BRAM" IP ROM mode.
    initial begin
        $readmemb("A_addr_pattern.mem", A_addr_pattern); // 9 signals * 14 rows
    end

    // select A_addr_pattern for the generated signal
    reg [ADDR_BIT_WIDTH*16*14-1:0] A_addr_pattern_for_signal;

    //************************** The synchronized always block below should only work in sliding state. **************************//

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) A_addr_pattern_for_signal <= {(ADDR_BIT_WIDTH*16*14){1'b0}};
        else A_addr_pattern_for_signal <= A_addr_pattern[zero_padding_signal];
    end // bram ROM requires one clk to read. -> this always block represent read only bram.
    //** at least change with state change to SLIDING

    reg [7:0] read_index_2; // 14*16 -> 8 bits required (Total clk : 14*16)

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) read_index_2 <= ADDR_BIT_WIDTH*14*16;
        else if(state == SLIDING) read_index_2 <= read_index_2 - 8'd7;
        else read_index_2 <= ADDR_BIT_WIDTH*14*16;
    end
    //** +1clk after SLIDING
    // Need clk cycle condition in SLIDING state to avoid read_index goes to negative number

    reg [9:0] addr_A, addr_A_current_tile;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) addr_A <= 10'd0;
        else if(state == SLIDING) addr_A <= addr_A_current_tile + A_addr_pattern_for_signal[read_index_2 -: ADDR_BIT_WIDTH];
        else if(state == DRAM_ACCESS) addr_A <= 10'd0;
        else if(state == KER_CHANGE) addr_A <= addr_A_current_tile;
        else addr_A <= addr_A;
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) addr_A_current_tile <= 10'd0;
        else if(state == TILE_CHANGE) addr_A_current_tile <= addr_A;
        else if(state == DRAM_ACCESS) addr_A_current_tile <= 10'd0;
        else addr_A_current_tile <= addr_A_current_tile;
    end



    //////////////////////////////
    ///      FOR MEMORY B      ///
    //////////////////////////////

    reg [ADDR_BIT_WIDTH*16*14-1:0] B_addr_pattern [8:0]; // max 16 clk * 14 rows * 9 signals

    // for the FPGA synthesis, below code be placed by .coe file initialization with "BRAM" IP ROM mode.
    initial begin
        $readmemb("B_addr_pattern.mem", B_addr_pattern); // 9 signals * 14 rows
    end

    // select B_addr_pattern for the generated signal
    reg [ADDR_BIT_WIDTH*16*14-1:0] B_addr_pattern_for_signal;

    //************************** The synchronized always block below should only work in sliding state. **************************//

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) B_addr_pattern_for_signal <= {(ADDR_BIT_WIDTH*16*14){1'b0}};
        else B_addr_pattern_for_signal <= B_addr_pattern[zero_padding_signal];
    end // bram ROM requires one clk to read. -> this always block represent read only bram.
    //** at least change with state change to SLIDING

    reg [9:0] addr_B, addr_B_current_tile;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) addr_B <= 10'd0;
        else if(state == SLIDING) addr_B <= addr_B_current_tile + B_addr_pattern_for_signal[read_index_2 -: ADDR_BIT_WIDTH];
        else if(state == DRAM_ACCESS) addr_B <= 10'd0;
        else if(state == KER_CHANGE) addr_B <= addr_B_current_tile;
        else addr_B <= addr_B;
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) addr_B_current_tile <= 10'd0;
        else if(state == TILE_CHANGE) addr_B_current_tile <= addr_B;
        else if(state == DRAM_ACCESS) addr_B_current_tile <= 10'd0;
        else addr_B_current_tile <= addr_B_current_tile;
    end


    //////////////////////////////
    ///      FOR MEMORY C      ///
    //////////////////////////////

    reg [ADDR_BIT_WIDTH*16*14-1:0] C_addr_pattern [8:0]; // max 16 clk * 14 rows * 9 signals

    // for the FPGA synthesis, below code be placed by .coe file initialization with "BRAM" IP ROM mode.
    initial begin
        $readmemb("C_addr_pattern.mem", C_addr_pattern); // 9 signals * 14 rows
    end

    // select B_addr_pattern for the generated signal
    reg [ADDR_BIT_WIDTH*16*14-1:0] C_addr_pattern_for_signal;

    //************************** The synchronized always block below should only work in sliding state. **************************//

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) C_addr_pattern_for_signal <= {(ADDR_BIT_WIDTH*16*14){1'b0}};
        else C_addr_pattern_for_signal <= C_addr_pattern[zero_padding_signal];
    end // bram ROM requires one clk to read. -> this always block represent read only bram.
    //** at least change with state change to SLIDING

    reg [9:0] addr_C, addr_C_current_tile;

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) addr_C <= 10'd0;
        else if(state == SLIDING) addr_C <= addr_C_current_tile + C_addr_pattern_for_signal[read_index_2 -: ADDR_BIT_WIDTH];
        else if(state == DRAM_ACCESS) addr_C <= 10'd0;
        else if(state == KER_CHANGE) addr_C <= addr_C_current_tile;
        else addr_B <= addr_B;
    end

    always @(posedge clk, negedge rst_n) begin
        if(!rst_n) addr_C_current_tile <= 10'd0;
        else if(state == TILE_CHANGE) addr_C_current_tile <= addr_B;
        else if(state == DRAM_ACCESS) addr_C_current_tile <= 10'd0;
        else addr_C_current_tile <= addr_C_current_tile;
    end
        
    //////////////////////////////////////// output signal assignment ////////////////////////////////////////

    // state transition (finish SLIDING) signal
    assign sliding_finish = (read_index_1 == 0) ? 1'b1 : 1'b0; // 1 clk signal

    // activation memory control signal
    assign addr_A_out = addr_A;
    assign addr_B_out = addr_B;
    assign addr_C_out = addr_C;

    assign en_A = (state == SLIDING) ? 1'b1 : 1'b0;
    assign en_B = (state == SLIDING) ? 1'b1 : 1'b0;
    assign en_C = (state == SLIDING) ? 1'b1 : 1'b0;

    // activation registor control signal
    assign act_mem_to_reg = (state == SLIDING) ? 1'b1 : 1'b0;

    endmodule

    
    

