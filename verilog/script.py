def generate_verilog_case():
    verilog_code = """always @(posedge clk, negedge rst_n) begin
    if (!rst_n) 
        before <= 16'd0;
    else if (start_accum) begin
        case (pixel_reg_idx)
"""
    
    # 0부터 195까지 Psum 인덱스 매칭
    for i in range(196):
        psum_index = (i + 1) % 196  # Psum 인덱스를 1씩 증가시키고 195 이후 0으로
        verilog_code += f"            8'd{i}: before <= Psum[{psum_index}];\n"
    
    verilog_code += """            default: before <= before;
        endcase
    end
end
"""
    return verilog_code

# Verilog 코드 출력
print(generate_verilog_case())
