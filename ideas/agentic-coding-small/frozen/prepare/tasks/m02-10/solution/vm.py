from assembler import assemble


def run(source: str, max_steps: int = 100) -> dict:
    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be positive")
    program = assemble(source)
    registers = dict.fromkeys(("a", "b", "c", "d"), 0)
    output = []
    pc = 0
    steps = 0
    halted = False
    while pc < len(program):
        if steps == max_steps:
            raise RuntimeError("instruction limit exceeded")
        instruction = program[pc]
        op = instruction[0]
        steps += 1
        next_pc = pc + 1
        if op == "SET":
            registers[instruction[1]] = instruction[2]
        elif op == "ADD":
            registers[instruction[1]] += instruction[2]
        elif op == "JMP":
            next_pc = instruction[1]
        elif op == "JNZ":
            if registers[instruction[1]] != 0:
                next_pc = instruction[2]
        elif op == "OUT":
            output.append(registers[instruction[1]])
        elif op == "HALT":
            halted = True
            break
        pc = next_pc
    return {
        "registers": registers,
        "output": output,
        "steps": steps,
        "halted": halted,
    }
