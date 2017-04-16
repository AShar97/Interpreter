""" Toy Language Interpreter """

""" Grammar :

	compound_statement : statement_list

	statement_list : statement
				   | statement SEMI statement_list

	statement : compound_statement
			  | assignment_statement
			  | loop_statement
			  | conditional_statement
			  | print_statement
			  | println_statement
			  | empty

	assignment_statement : variable ASSIGN expr

	loop_statement : WHILE cond_expr DO compound_statement DONE

	conditional_statement : IF cond_expr THEN compound_statement ELSE compound_statement END

	print_statement : PRINT expr

	println_statement : PRINTLN

	empty :

	cond_expr : ???

	expr : string ???
		 | term ((PLUS | MINUS) term)*

	term : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*

	factor : PLUS factor
		   | MINUS factor
		   | INTEGER_CONST
		   | REAL_CONST
		   | LPAREN expr RPAREN
		   | variable

	variable: ID

"""

from collections import OrderedDict

###############################################################################
#                                                                             #
#  LEXER                                                                      #
#                                                                             #
###############################################################################

# Token types
#
# EOF (end-of-file) token is used to indicate that
# there is no more input left for lexical analysis

INTEGER_CONST, REAL_CONST = ('INTEGER_CONST', 'REAL_CONST')
#STRING = 'STRING'
#QUOTE = 'QUOTE'

PLUS, MINUS, MUL, INTEGER_DIV, FLOAT_DIV = ('PLUS', 'MINUS', 'MUL', 'INTEGER_DIV', 'FLOAT_DIV')

LPAREN, RPAREN = ('LPAREN', 'RPAREN')

ID = 'ID'

ASSIGN = 'ASSIGN'

SEMI = 'SEMI'

ET, NET, LET, GET, LT, GT = ('ET', 'NET', 'LET', 'GET', 'LT', 'GT')

AND, OR = ('AND', 'OR')

WHILE, DO, DONE = ('WHILE', 'DO', 'DONE')

IF, THEN, ELSE, END = ('IF', 'THEN', 'ELSE', 'END')

PRINT = 'PRINT'
PRINTLN = 'PRINTLN'

EOF = 'EOF'


class Token(object):
	def __init__(self, type, value):
		self.type = type
		self.value = value

	def __str__(self):
		"""String representation of the class instance.

		Examples:
			Token(INTEGER, 3)
			Token(PLUS '+')
			Token(MUL, '*')
		"""
		return 'Token({type}, {value})'.format(
			type = self.type,
			value = repr(self.value)
		)

	def __repr__(self):
		return self.__str__()

RESERVED_KEYWORDS = {
	'IF': Token('IF', 'IF'),
	'THEN': Token('THEN', 'THEN'),
	'ELSE': Token('ELSE', 'ELSE'),
	'END': Token('END', 'END'),

	'WHILE': Token('WHILE', 'WHILE'),
	'DO': Token('DO', 'DO'),
	'DONE': Token('DONE', 'DONE'),

	'PRINT': Token('PRINT', 'PRINT'),
	'PRINTLN': Token('PRINTLN', 'PRINTLN'),
}


#Breaking into series of tokens.
class Lexer(object):
	def __init__(self, text):
		# client string input, e.g. "4 + 2 * 3 - 6 / 2"
		self.text = text
		# self.pos is an index into self.text
		self.pos = 0
		self.current_char = self.text[self.pos]

	def error(self):
		raise Exception('Invalid character')

	def advance(self):
		"""Advance the 'pos' pointer and set the 'current_char' variable."""
		self.pos += 1
		if self.pos > len(self.text) - 1:
			self.current_char = None  # Indicates end of input
		else:
			self.current_char = self.text[self.pos]

	def peek(self):
		peek_pos = self.pos + 1
		if peek_pos > len(self.text) - 1:
			return None
		else:
			return self.text[peek_pos]

	def skip_comment(self):
		while self.current_char != '}':
			self.advance()
		self.advance()  # the closing curly brace

	def skip_whitespace(self):
		while self.current_char is not None and self.current_char.isspace():
			self.advance()

	def string(self):
		"""Return a string consumed from the input."""
		result = ''
		while self.current_char != '\"':
			result += self.current_char
			self.advance()
		self.advance()	# the ending '\"' symbol

		token = Token('STRING', result)
		return token

	def number(self):
		"""Return a (multidigit) integer or float consumed from the input."""
		result = ''
		while self.current_char is not None and self.current_char.isdigit():
			result += self.current_char
			self.advance()

		if self.current_char == '.':
			result += self.current_char
			self.advance()

			while (
				self.current_char is not None and
				self.current_char.isdigit()
			):
				result += self.current_char
				self.advance()

			token = Token('REAL_CONST', float(result))
		else:
			token = Token('INTEGER_CONST', int(result))

		return token

	def _id(self):
		"""Handle identifiers and reserved keywords"""
		result = ''
		while self.current_char is not None and self.current_char.isalnum():
			result += self.current_char
			self.advance()

		token = RESERVED_KEYWORDS.get(result, Token(ID, result))
		return token

	def get_next_token(self):
		"""Lexical analyzer (also known as scanner or tokenizer)

		This method is responsible for breaking a sentence
		apart into tokens. One token at a time.
		"""
		while self.current_char is not None:

			if self.current_char == '{':
				self.advance()
				self.skip_comment()
				continue

#			if self.current_char == '\"':
#				self.advance()
##				self.string()
##				continue
#				return Token(QUOTE, '\"')

			if self.current_char.isspace():
				self.skip_whitespace()
				continue

			if self.current_char.isdigit():
				return self.number()

			if self.current_char.isalpha():
				return self._id()

			if self.current_char == ':' and self.peek() == '=':
				self.advance()
				self.advance()
				return Token(ASSIGN, ':=')

			if self.current_char == '=' and self.peek() == '=':
				self.advance()
				self.advance()
				#token=Token(ET, '==')
				#return CondET(token)
				return Token(ET, '==')
			#	return Cond(Token(ET, '=='))

			if self.current_char == '!' and self.peek() == '=':
				self.advance()
				self.advance()
				#token=Token(NET, '!=')
				#return CondNET(token)
				return Token(NET, '!=')
			#	return Cond(Token(NET, '!='))

			if self.current_char == '<' and self.peek() == '=':
				self.advance()
				self.advance()
				#token=Token(LET, '<=')
				#return CondLET(token)
				return Token(LET, '<=')
			#	return Cond(Token(LET, '<='))

			if self.current_char == '>' and self.peek() == '=':
				self.advance()
				self.advance()
				#token=Token(GET, '>=')
				#return CondGET(token)
				return Token(GET, '>=')
			#	return Cond(Token(GET, '>='))

			if self.current_char == '<' :
				self.advance()
				
				#token=Token(LT, '<')
				#return CondLT(token)
				return Token(LT, '<')
			#	return Cond(Token(LT, '<'))

			if self.current_char == '>' :
				self.advance()
				
				#token=Token(GT, '>')
				#return CondGT(token)
				return Token(GT, '>')
			#	return Cond(Token(GT, '>'))

			if self.current_char == '&' and self.peek() == '&':
				self.advance()
				self.advance()
				return Token(AND, '&&')

			if self.current_char == '|' and self.peek() == '|':
				self.advance()
				self.advance()
				return Token(OR, '||')

			if self.current_char == ';':
				self.advance()
				return Token(SEMI, ';')

			if self.current_char == '+':
				self.advance()
				return Token(PLUS, '+')

			if self.current_char == '-':
				self.advance()
				return Token(MINUS, '-')

			if self.current_char == '*':
				self.advance()
				return Token(MUL, '*')

			if self.current_char == '/' and self.peek() == '/':
				self.advance()
				self.advance()
				return Token(INTEGER_DIV, '//')

			if self.current_char == '/':
				self.advance()
				return Token(FLOAT_DIV, '/')

			if self.current_char == '(':
				self.advance()
				return Token(LPAREN, '(')

			if self.current_char == ')':
				self.advance()
				return Token(RPAREN, ')')

			self.error()

		return Token(EOF, None)


###############################################################################
#                                                                             #
#  PARSER                                                                     #
#                                                                             #
###############################################################################

# abstract-syntax trees
class AST(object):
	pass


class Compound(AST):
	"""Represents a '...' block"""
	def __init__(self):
		self.children = []


class Assign(AST):
	def __init__(self, left, op, right):
		self.left = left
		self.token = self.op = op
		self.right = right


class Num(AST):
	def __init__(self, token):
		self.token = token
		self.value = token.value


#class String(AST):
#	def __init__(self, token):
#		self.token = token
#		self.value = token.value


class Var(AST):
	"""The Var node is constructed out of ID token."""
	def __init__(self, token):
		self.token = token
		self.value = token.value


class NoOp(AST):
	pass


class UnaryOp(AST):
	def __init__(self, op, expr):
		self.token = self.op = op
		self.expr = expr


class BinOp(AST):
	def __init__(self, left, op, right):
		self.left = left
		self.token = self.op = op
		self.right = right


class LogOp(AST):
	def __init__(self, left, op, right):
		self.left = left
		self.token = self.op = op
		self.right = right


class CondOp(AST):
	def __init__(self, left, op, right):
		self.left = left
		self.token = self.op = op
		self.type = op.type
		self.right = right

#class Cond(AST):
#	def __init__(self, op):
#		self.left = None
#		self.token = self.op = op
#		self.type = op.type
#		self.right = None

class Loop(AST):
	def __init__(self, cond, stmt):
		self.cond = cond
		self.stmt = stmt


class Conditional(AST):
	def __init__(self, cond, then_stmt, else_stmt):
		self.cond = cond
		self.then_stmt = then_stmt
		self.else_stmt = else_stmt


class PrintLN(AST):
	def __init__(self):
		pass


class Print(AST):
	def __init__(self, expr):
		self.data = expr


class Parser(object):
##class Parser(Lexer):
	def __init__(self, lexer):
	##def __init__(self, text):
		self.lexer = lexer
		##super(Parser,self).__init__(text)
		# set current token to the first token taken from the input
		self.current_token = self.lexer.get_next_token()
		##self.current_token = self.get_next_token()

	def error(self):
		raise Exception('Invalid syntax')

	def eat(self, token_type):
		# compare the current token type with the passed token
		# type and if they match then "eat" the current token
		# and assign the next token to the self.current_token,
		# otherwise raise an exception.
		if self.current_token.type == token_type:
			self.current_token = self.lexer.get_next_token()
			##self.current_token = self.get_next_token()
		else:
			self.error()

	def compound_statement(self):
		"""
		compound_statement: statement_list
		"""
		nodes = self.statement_list()

		root = Compound()
		for node in nodes:
			root.children.append(node)

		return root

	def statement_list(self):
		"""
		statement_list : statement
					   | statement SEMI statement_list
		"""
		node = self.statement()

		results = [node]

		while self.current_token.type == SEMI:
			self.eat(SEMI)
			results.append(self.statement())

		if self.current_token.type == ID:
			self.error()

		return results

	def statement(self):
		"""
		type of statement.
		"""
		if self.current_token.type == ID:
			node = self.assignment_statement()
		elif self.current_token.type == WHILE:
			node=self.loop_statement()
		elif self.current_token.type == IF:
			node=self.conditional_statement()
		elif self.current_token.type == PRINTLN:
			node=self.println_statement()
		elif self.current_token.type == PRINT:
			node=self.print_statement()
		else:
			node = self.empty()
		return node

	def assignment_statement(self):
		"""
		assignment_statement : variable ASSIGN expr
		"""
		left = self.variable()
		token = self.current_token
		self.eat(ASSIGN)
		right = self.expr()
		node = Assign(left, token, right)
		return node

	def loop_statement(self):
		"""
		loop_statement : WHILE cond_expr DO compound_statement DONE
		"""
		self.eat(WHILE)
		cond=self.cond_expr()
		self.eat(DO)
		stmt=self.compound_statement()
		self.eat(DONE)
		node = Loop(cond,stmt)
		return node

	def conditional_statement(self):
		"""
		conditional_statement : IF cond_expr THEN compound_statement ELSE compound_statement END
		"""
		self.eat(IF)
		cond=self.cond_expr()
		self.eat(THEN)
		then_stmt=self.compound_statement()
		self.eat(ELSE)
		else_stmt=self.compound_statement()
		self.eat(END)
		node = Conditional(cond,then_stmt,else_stmt)
		
		return node

	def println_statement(self):
		"""
		println_statement : PRINTLN
		"""
		self.eat(PRINTLN)
		node = PrintLN()
		return node

	def print_statement(self):
		"""
		print_statement : PRINT expr
		"""
		self.eat(PRINT)
		data = self.expr()
		node = Print(data)
		return node

	def cond_expr(self):
		
		node = self.cond_term()

		while self.current_token.type==OR:
			token = self.current_token
			self.eat(OR)

			node = LogOp(left=node, op=token, right=self.cond_term())

		return node

	def cond_term(self):
		
		node = self.cond_factor()

		while self.current_token.type==AND:
			token = self.current_token
			self.eat(AND)

			node = LogOp(left=node, op=token, right=self.cond_factor())

		return node

	def cond_factor(self):
		
		left = self.expr()
#		node = self.current_token
		token = self.current_token

		##self.current_token=self.get_next_token()
		self.current_token=self.lexer.get_next_token()
		right=self.expr()
#		node.left=left
#		node.right=right
		node = CondOp(left=left, op=token, right=right)
		return node

	def variable(self):
		"""
		variable : ID
		"""
		node = Var(self.current_token)
		self.eat(ID)
		return node

	def empty(self):
		"""An empty production"""
		return NoOp()

	def factor(self):
		"""factor : PLUS factor
				  | MINUS factor
				  | INTEGER_CONST
				  | REAL_CONST
				  | LPAREN expr RPAREN
				  | variable
		"""
		token = self.current_token
		if token.type == PLUS:
			self.eat(PLUS)
			node = UnaryOp(token, self.factor())
			return node
		elif token.type == MINUS:
			self.eat(MINUS)
			node = UnaryOp(token, self.factor())
			return node
		###
#		elif token.type == QUOTE:
#			self.eat(QUOTE)
#			node = self.string()
#			node = self.expr()
#			self.eat(QUOTE)
#			return node
		elif token.type == INTEGER_CONST:
			self.eat(INTEGER_CONST)
			return Num(token)
		elif token.type == REAL_CONST:
			self.eat(REAL_CONST)
			return Num(token)
		elif token.type == LPAREN:
			self.eat(LPAREN)
			node = self.expr()
			self.eat(RPAREN)
			return node
		else:
			node = self.variable()
			return node

	def term(self):
		"""term : factor ((MUL | INTEGER_DIV | FLOAT_DIV) factor)*"""
		node = self.factor()

		while self.current_token.type in (MUL, INTEGER_DIV, FLOAT_DIV):
			token = self.current_token
			if token.type == MUL:
				self.eat(MUL)
			elif token.type == INTEGER_DIV:
				self.eat(INTEGER_DIV)
			elif token.type == FLOAT_DIV:
				self.eat(FLOAT_DIV)

			node = BinOp(left=node, op=token, right=self.factor())

		return node

	def expr(self):
		"""
		expr   : string ???
			   | term ((PLUS | MINUS) term)*
		term   : factor ((MUL | DIV) factor)*
		factor : PLUS factor
			   | MINUS factor
			   | INTEGER_CONST
			   | REAL_CONST
			   | LPAREN expr RPAREN
			   | variable
		"""
#		if self.current_token.type == STRING:
#			self.eat(STRING)
#			return String(self.current_token)

#		else:

		node = self.term()

		while self.current_token.type in (PLUS, MINUS):
			token = self.current_token
			if token.type == PLUS:
				self.eat(PLUS)
			elif token.type == MINUS:
				self.eat(MINUS)

			node = BinOp(left=node, op=token, right=self.term())

		return node

	def parse(self):
		node = self.compound_statement()
		if self.current_token.type != EOF:
			self.error()

		return node


###############################################################################
#                                                                             #
#  INTERPRETER                                                                #
#                                                                             #
###############################################################################

class NodeVisitor(object):
	def visit(self, node):
		method_name = 'visit_' + type(node).__name__
		visitor = getattr(self, method_name, self.generic_visit)
		return visitor(node)

	def generic_visit(self, node):
		raise Exception('No visit_{} method'.format(type(node).__name__))


class Interpreter(NodeVisitor):

	GLOBAL_SCOPE = OrderedDict()

	def __init__(self, parser):
		self.parser = parser

	def visit_Compound(self, node):
		for child in node.children:
			self.visit(child)

	def visit_NoOp(self, node):
		pass

	def visit_Assign(self, node):
		var_name = node.left.value
		self.GLOBAL_SCOPE[var_name] = self.visit(node.right)

	def visit_Var(self, node):
		var_name = node.value
		val = self.GLOBAL_SCOPE.get(var_name)
		if val is None:
			raise NameError(repr(var_name))
		else:
			return val

	def visit_UnaryOp(self, node):
		op = node.op.type
		if op == PLUS:
			return +self.visit(node.expr)
		elif op == MINUS:
			return -self.visit(node.expr)

	def visit_BinOp(self, node):
		if node.op.type == PLUS:
			return self.visit(node.left) + self.visit(node.right)
		elif node.op.type == MINUS:
			return self.visit(node.left) - self.visit(node.right)
		elif node.op.type == MUL:
			return self.visit(node.left) * self.visit(node.right)
		elif node.op.type == INTEGER_DIV:
			return self.visit(node.left) // self.visit(node.right)
		elif node.op.type == FLOAT_DIV:
			return float(self.visit(node.left)) / float(self.visit(node.right))

	def visit_LogOp(self, node):
		if node.op.type == AND:
			return self.visit(node.left) and self.visit(node.right)
		elif node.op.type == OR:
			return self.visit(node.left) or self.visit(node.right)

	def visit_CondOp(self, node):
		if node.op.type == ET:
			return self.visit(node.left) == self.visit(node.right)
		elif node.op.type == NET:
			return self.visit(node.left) != self.visit(node.right)
		elif node.op.type == GET:
			return self.visit(node.left) >= self.visit(node.right)
		elif node.op.type == LET:
			return self.visit(node.left) <= self.visit(node.right)
		elif node.op.type == GT:
			return self.visit(node.left) > self.visit(node.right)
		elif node.op.type == LT:
			return self.visit(node.left) < self.visit(node.right)
#	def visit_Cond(self, node):
#		if node.op.type == ET:
#			return self.visit(node.left) == self.visit(node.right)
#		elif node.op.type == NET:
#			return self.visit(node.left) != self.visit(node.right)
#		elif node.op.type == GET:
#			return self.visit(node.left) >= self.visit(node.right)
#		elif node.op.type == LET:
#			return self.visit(node.left) <= self.visit(node.right)
#		elif node.op.type == GT:
#			return self.visit(node.left) > self.visit(node.right)
#		elif node.op.type == LT:
#			return self.visit(node.left) < self.visit(node.right)

	def visit_Loop(self, node):
		while(self.visit(node.cond) == True):
			self.visit(node.stmt)

		##if(self.cond.eval() ==True):
		##	self.stmt.eval()
		##	self.eval()

	def visit_Conditional(self, node):
		if(self.visit(node.cond) == True):
			self.visit(node.then_stmt)
		else:
			self.visit(node.else_stmt)

	def visit_PrintLN(self, node):
		print("\n", end="")

	def visit_Print(self, node):
		print(self.visit(node.data), end="")

	def visit_Num(self, node):
		return node.value

	def visit_String(self, node):
		return node.value

	def interpret(self):
		tree = self.parser.parse()
		return self.visit(tree)


def main():
	import sys
	text = open(sys.argv[1], 'r').read()
		
	lexer = Lexer(text)
	parser = Parser(lexer)
	##parser = Parser(text)
	interpreter = Interpreter(parser)
	result = interpreter.interpret()

#	for k, v in sorted(interpreter.GLOBAL_SCOPE.items()):
#		print('%s = %s' % (k, v))


if __name__ == '__main__':
	main()
	