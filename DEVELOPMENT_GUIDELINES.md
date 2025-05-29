# Development Guidelines - Manual Enforcement

## Core Principles
- NO direct pushes to main branch
- ALL changes go through pull requests
- ALL pull requests require partner review and approval
- NO self-merging of pull requests

## Workflow Process

### 1. Starting New Work
git checkout main
git pull origin main
git checkout -b feature/your-feature-name

### 2. Making Changes
- Work in small, focused commits
- Write descriptive commit messages
- Test your changes locally before pushing

### 3. Creating Pull Request
git push origin feature/your-feature-name

- Create PR on GitHub
- Write clear title and description
- Tag your partner for review
- Wait for approval before merging

### 4. Code Review Checklist
**For the Reviewer:**
- [ ] Code compiles and runs without errors
- [ ] Changes align with project requirements
- [ ] Code follows agreed coding standards
- [ ] No obvious bugs or security issues
- [ ] All conversations resolved before approval

**For the Author:**
- [ ] Respond to all feedback
- [ ] Make requested changes
- [ ] Re-request review after changes

### 5. Merging Rules
- Only merge after explicit approval from partner
- Delete feature branch after successful merge
- Pull latest main before starting new work

## Emergency Procedures
If urgent fixes are needed:
- Still create a pull request
- Review can be expedited but NOT skipped
- Document the urgency in PR description

## Pull Request Best Practices

### Size and Scope
- Keep PRs **focused and atomic** - one feature/fix per PR
- Target **50-200 lines of changed code** when possible
- Break large features into smaller, reviewable chunks

### PR Title and Description Standards
- Use clear, descriptive titles that explain the change
- Follow format: `type(scope): description`
  - Examples: `feat(auth): add user login validation`
  - Examples: `fix(ui): resolve button alignment issue`

### Required PR Description Template
What This PR Does
[Brief description of changes]

Why This Change Is Needed
[Problem being solved]

How to Test
[Steps to verify the changes work]

Potential Risks/Considerations
[Any side effects or concerns]

### Commit Message Standards
- Use **Conventional Commits Specification**:
  - `feat: add new feature`
  - `fix: resolve bug`
  - `docs: update documentation`
  - `test: add unit tests`
- Avoid vague messages like "fixed stuff" or "updated code"

### Code Review Process
- **Review your own PR first** before requesting partner review
- Provide **constructive feedback** with suggested solutions
- Use labels: 'bug fix,' 'feature,' 'critical' to categorize PRs
- **Address ALL review comments** before merging
- Write "done" in response to comments (don't mark as resolved yourself)

### Merging Rules
- Only merge after explicit approval from partner
- **Never force push** (`git push --force`)
- Delete feature branch after successful merge