from mcp.server.fastmcp import FastMCP

mcp = FastMCP("BMI")

# Tools
@mcp.tool()
def calculate_bmi(weight: int, height: int) -> str:
    """Calculate BMI"""
    return "BMI: "+str(weight/(height*height))

if __name__ == "__main__":
    mcp.run(transport="streamable-http")