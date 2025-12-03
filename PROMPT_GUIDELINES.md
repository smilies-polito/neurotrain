# Prompt Guidelines for Feature Implementation

> Best practices for writing prompts that help LLMs implement new features effectively.  
> Applicable to any software project.

---

## 1. Core Principles

### 1.1 Be Explicit, Not Implicit

LLMs work best when constraints are **stated**, not assumed.

```
❌ BAD:  "Add user authentication"
✅ GOOD: "Add user authentication using JWT tokens, storing refresh tokens 
         in httpOnly cookies, with a 15-minute access token expiry"
```

### 1.2 Provide Context Before Requirements

LLMs need to understand the **current state** before making changes.

```
❌ BAD:  "Add a new API endpoint"
✅ GOOD: "Given the existing Express app structure in src/routes/, 
         add a new API endpoint following the pattern in users.route.ts"
```

### 1.3 Constrain to Prevent Over-Engineering

Without explicit limits, LLMs tend to add unnecessary complexity.

```
❌ BAD:  "Improve error handling"
✅ GOOD: "Add try-catch to the database calls in userService.ts. 
         Return 500 status with generic message. Do NOT add custom error classes."
```

---

## 2. Prompt Structure Template

```markdown
# [Feature Name]

## Context
- Current state: [reference existing files/docs]
- Pattern to follow: [specific file as example]
- Related components: [what this interacts with]

## Objective
[One clear sentence describing what to build]

## Requirements
1. [Specific requirement]
2. [Specific requirement]
...

## Deliverables
- [ ] `path/to/file.ext` (~N lines) - [purpose]
- [ ] `path/to/file.ext` (modify) - [what to change]
...

## Constraints
- Use [library/pattern] for [functionality]
- Do NOT [specific anti-pattern]
- Keep [scope limitation]

## Validation
- [Measurable success criterion]
- [Measurable success criterion]

## Out of Scope
- [What NOT to do in this prompt]
```

---

## 3. Context Section Best Practices

### 3.1 Reference Existing Code

Always point to existing patterns:

```markdown
## Context
- Follow the pattern in `src/services/userService.ts`
- Use the existing `DatabaseConnection` class from `src/db/connection.ts`
- Integrate with the auth middleware in `src/middleware/auth.ts`
```

### 3.2 State the Current Architecture

```markdown
## Context
Current structure:
```
src/
├── controllers/    # HTTP handlers
├── services/       # Business logic
├── repositories/   # Data access
└── models/         # Type definitions
```
New code should follow this layered architecture.
```

### 3.3 Link to Documentation

```markdown
## Context
- Reference: PROJECT_STATE.md for current implementation status
- Reference: docs/API.md for endpoint conventions
- Reference: CONTRIBUTING.md for code style
```

---

## 4. Requirements Section Best Practices

### 4.1 Use Numbered Lists

Easier for LLMs to track and verify:

```markdown
## Requirements
1. Create endpoint POST /api/users
2. Validate email format using regex
3. Hash password with bcrypt (10 rounds)
4. Return 201 with user object (exclude password)
5. Return 400 if email already exists
```

### 4.2 Specify Interfaces/Types

```markdown
## Requirements
Input schema:
```typescript
interface CreateUserRequest {
  email: string;      // required, valid email
  password: string;   // required, min 8 chars
  name?: string;      // optional
}
```

Output schema:
```typescript
interface UserResponse {
  id: string;
  email: string;
  name: string | null;
  createdAt: string;  // ISO 8601
}
```
```

### 4.3 Define Behavior for Edge Cases

```markdown
## Requirements
Error handling:
- Invalid email format → 400 `{ "error": "Invalid email format" }`
- Email exists → 409 `{ "error": "Email already registered" }`
- Database error → 500 `{ "error": "Internal server error" }` (log details)
```

---

## 5. Constraints Section Best Practices

### 5.1 Library Usage (Reuse Over Reimplement)

```markdown
## Constraints
USE existing libraries:
- Validation: use `zod` (already in dependencies)
- Password hashing: use `bcrypt` (already in dependencies)
- JWT: use `jsonwebtoken` (already in dependencies)

DO NOT implement:
- Custom validation logic
- Custom hashing algorithms
- Custom token generation
```

### 5.2 Scope Limitations

```markdown
## Constraints
Scope:
- Only modify files in src/auth/
- Do not change database schema
- Do not add new dependencies
- Maximum 200 lines of new code
```

### 5.3 Pattern Enforcement

```markdown
## Constraints
Patterns:
- Use async/await (no callbacks, no .then())
- Use dependency injection (no direct imports of singletons)
- Use early returns (no deep nesting)
- Keep functions under 30 lines
```

---

## 6. Deliverables Section Best Practices

### 6.1 Be Explicit About Files

```markdown
## Deliverables
New files:
- [ ] `src/auth/authController.ts` (~50 lines) - HTTP handlers
- [ ] `src/auth/authService.ts` (~80 lines) - business logic
- [ ] `src/auth/authTypes.ts` (~20 lines) - TypeScript interfaces

Modified files:
- [ ] `src/routes/index.ts` - add auth routes import
- [ ] `src/middleware/index.ts` - export new auth middleware

Test files:
- [ ] `tests/auth/authService.test.ts` (~60 lines) - unit tests
```

### 6.2 Estimate Complexity

| Scope | Files | Lines | Appropriate For |
|-------|-------|-------|-----------------|
| Tiny | 1 | <50 | Bug fix, small utility |
| Small | 1-2 | 50-150 | Single feature |
| Medium | 3-5 | 150-400 | Feature with tests |
| Large | 5+ | 400+ | **Split into multiple prompts** |

### 6.3 Specify Test Requirements

```markdown
## Deliverables
Tests must cover:
- [ ] Happy path (valid input → expected output)
- [ ] Validation errors (invalid input → 400)
- [ ] Not found cases (missing resource → 404)
- [ ] Auth failures (invalid token → 401)
```

---

## 7. Validation Section Best Practices

### 7.1 Measurable Criteria

```markdown
## Validation
✅ Success criteria:
- All tests pass: `npm test src/auth/`
- No TypeScript errors: `npm run typecheck`
- Linting passes: `npm run lint`
- Endpoint responds correctly: `curl -X POST localhost:3000/api/auth/login`
```

### 7.2 Performance Expectations

```markdown
## Validation
Performance:
- Response time < 200ms for login endpoint
- Memory usage does not increase by more than 10MB
- No N+1 query patterns
```

### 7.3 Integration Verification

```markdown
## Validation
Integration:
- Works with existing user service
- Compatible with current session middleware
- Does not break existing tests
```

---

## 8. Anti-Patterns to Avoid

### 8.1 Vague Requirements

```markdown
❌ BAD:
"Make the API better"

✅ GOOD:
"Add request validation to POST /api/users using zod schema.
Return 400 with field-specific error messages for invalid input."
```

### 8.2 Missing Context

```markdown
❌ BAD:
"Add caching"

✅ GOOD:
"Add Redis caching to getUserById() in src/services/userService.ts.
Cache for 5 minutes. Use the existing Redis client from src/cache/redis.ts.
Invalidate cache on user update."
```

### 8.3 Unbounded Scope

```markdown
❌ BAD:
"Implement user management"

✅ GOOD:
"Implement user creation endpoint only.
Out of scope: user update, delete, list, password reset.
Those will be separate prompts."
```

### 8.4 No Negative Constraints

```markdown
❌ BAD:
"Add logging"

✅ GOOD:
"Add structured logging using winston.
Log: request ID, user ID, action, timestamp.
Do NOT log: passwords, tokens, PII, full request bodies.
Do NOT add console.log statements."
```

---

## 9. Prompt Size Guidelines

### 9.1 Keep Prompts Focused

| Prompt Type | Ideal Length | Contains |
|-------------|--------------|----------|
| Micro | 5-10 lines | Single function change |
| Standard | 20-50 lines | Single feature |
| Detailed | 50-100 lines | Feature with edge cases |
| Epic | 100+ lines | **Should be split** |

### 9.2 When to Split

Split into multiple prompts when:
- More than 5 files need changes
- Multiple unrelated components involved
- Feature has distinct phases (backend → frontend → tests)
- Total estimated lines > 400

### 9.3 Sequential Prompts

```markdown
# Prompt 1: Data Layer
Add UserRepository with CRUD operations.
Deliverables: src/repositories/userRepository.ts, tests

# Prompt 2: Service Layer  
Add UserService using UserRepository from Prompt 1.
Deliverables: src/services/userService.ts, tests

# Prompt 3: API Layer
Add user endpoints using UserService from Prompt 2.
Deliverables: src/controllers/userController.ts, routes, tests
```

---

## 10. Language/Framework Specific Tips

### 10.1 For TypeScript/JavaScript

```markdown
## Constraints
- Use strict TypeScript (no `any` types)
- Export types from dedicated `.types.ts` files
- Use named exports (no default exports)
- Prefer interfaces over type aliases for objects
```

### 10.2 For Python

```markdown
## Constraints
- Use type hints for all function signatures
- Use dataclasses for data structures
- Follow PEP 8 (enforced by black, line-length=88)
- Use pytest fixtures for test setup
```

### 10.3 For API Development

```markdown
## Constraints
- Use RESTful conventions (nouns for resources, HTTP verbs for actions)
- Return appropriate status codes (201 for create, 204 for delete)
- Include request ID in all responses
- Document with OpenAPI/Swagger comments
```

---

## 11. Testing Prompt Template

```markdown
# Add Tests for [Component]

## Context
- Component: `src/services/userService.ts`
- Test framework: Jest with ts-jest
- Existing tests: `tests/services/*.test.ts`

## Requirements
Test cases:
1. [Function]: returns expected output for valid input
2. [Function]: throws [ErrorType] for invalid input
3. [Function]: handles empty/null input gracefully
4. [Function]: integrates correctly with [Dependency]

## Constraints
- Use existing test utilities from `tests/helpers/`
- Mock external dependencies (database, APIs)
- No integration tests (unit only)
- Each test < 20 lines

## Deliverables
- [ ] `tests/services/userService.test.ts` (~100 lines)

## Validation
- All tests pass: `npm test -- userService`
- Coverage > 80% for userService.ts
```

---

## 12. Refactoring Prompt Template

```markdown
# Refactor [Component]

## Context
- Current: `src/legacy/oldComponent.ts` (problematic)
- Target pattern: `src/components/goodExample.ts`
- Reason: [why refactoring is needed]

## Requirements
1. Extract [X] into separate function/class
2. Replace [old pattern] with [new pattern]
3. Maintain backward compatibility (same public API)
4. Add missing type annotations

## Constraints
- Do NOT change public function signatures
- Do NOT change behavior (refactor only)
- Do NOT fix unrelated issues
- Keep commits atomic (one logical change each)

## Validation
- Existing tests still pass
- No TypeScript errors
- Behavior unchanged (verified by tests)

## Out of Scope
- Performance optimization
- New features
- Dependency updates
```

---

## 13. Quick Reference Checklist

Before sending a prompt, verify:

### Context
- [ ] Referenced existing files/patterns
- [ ] Stated current project structure
- [ ] Linked relevant documentation

### Requirements  
- [ ] Used numbered list
- [ ] Specified input/output types
- [ ] Defined edge case behavior

### Constraints
- [ ] Listed libraries to use
- [ ] Listed what NOT to do
- [ ] Set scope boundaries

### Deliverables
- [ ] Listed all files (new and modified)
- [ ] Estimated line counts
- [ ] Included test files

### Validation
- [ ] Defined measurable success criteria
- [ ] Specified how to verify

### Scope
- [ ] Single focused feature
- [ ] < 5 files, < 400 lines
- [ ] Out of scope section if needed

---

## 14. Example: Complete Well-Formed Prompt

```markdown
# Add Password Reset Flow

## Context
- Auth system: `src/auth/` (JWT-based, existing)
- Email service: `src/services/emailService.ts` (existing)
- User model: `src/models/User.ts` (has email, passwordHash fields)
- Pattern: follow `src/auth/loginController.ts`

## Objective
Allow users to reset their password via email link.

## Requirements
1. POST /api/auth/forgot-password
   - Input: { email: string }
   - Generate 6-digit code, store with 15-min expiry
   - Send code via emailService.sendPasswordReset()
   - Return 200 (same response whether email exists or not)

2. POST /api/auth/reset-password
   - Input: { email: string, code: string, newPassword: string }
   - Validate code and expiry
   - Hash new password with bcrypt (10 rounds)
   - Invalidate code after use
   - Return 200 on success, 400 on invalid/expired code

## Deliverables
- [ ] `src/auth/passwordResetController.ts` (~60 lines)
- [ ] `src/auth/passwordResetService.ts` (~80 lines)
- [ ] `src/models/PasswordResetCode.ts` (~15 lines)
- [ ] `tests/auth/passwordReset.test.ts` (~100 lines)
- [ ] `src/routes/auth.ts` (modify) - add new routes

## Constraints
- Use existing emailService (do not create new email logic)
- Use existing bcrypt utility from auth/utils.ts
- Store codes in memory (Map) for now, not database
- Do NOT add rate limiting (separate feature)

## Validation
- Tests pass for happy path and error cases
- Code expires after 15 minutes
- Same response for existing/non-existing email (security)
- Password actually changes in user record

## Out of Scope
- Rate limiting
- Account lockout
- Password strength validation (exists elsewhere)
- Email templates (use plain text)
```
