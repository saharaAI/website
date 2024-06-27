from marshmallow import Schema, fields, validate

class PromptRequestSchema(Schema):
    number = fields.Integer(required=True)
    context = fields.String(required=False)

class GenerateRequestSchema(Schema):
    message = fields.String(required=True)
    prompt = fields.Nested(PromptRequestSchema, required=True)