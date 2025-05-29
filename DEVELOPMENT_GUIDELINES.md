# Development Guidelines - Manual Enforcement

## Core Principles
- NO direct pushes to main branch
- ALL changes go through pull requests
- ALL pull requests require partner review and approval
- NO self-merging of pull requests

## Branch Naming Convention
Use descriptive branch names that indicate the type and purpose of work:

**Required Format**: `type/descriptive-name`

**Examples:**
- `feature/poker-game-logic`
- `feature/user-authentication`
- `feature/cfr-algorithm-implementation`
- `bugfix/ui-rendering-issue`
- `bugfix/login-button-crash`
- `hotfix/critical-security-patch`
- `docs/api-documentation`
- `test/unit-test-coverage`

**Naming Rules:**
- Use lowercase with hyphens (kebab-case)
- Be descriptive but concise
- Include the feature/component being worked on
- Avoid personal identifiers (use purpose, not "john-feature")

## Workflow Process

### 1. Starting New Work
```
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

### 2. Making Changes
- Work in small, focused commits
- Write descriptive commit messages
- Test your changes locally before pushing

### 3. Creating Pull Request
```
git push origin feature/your-feature-name
```

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

## Coding Standards

### JavaScript/React Standards
- Use **consistent indentation** (2 spaces)
- Use **meaningful variable names** (no single letters except for loops)
- Add **comments for complex logic**
- Follow **React naming conventions** (PascalCase for components)
- Use **ESLint** recommendations when available

### File Organization
```
src/
├── components/ # Reusable UI components
├── pages/ # Route-specific components
├── utils/ # Helper functions
├── hooks/ # Custom React hooks
├── styles/ # CSS/styling files
└── tests/ # Test files
```

## Testing Requirements

### Before Creating PR
- [ ] Run `npm run dev` - ensure app starts without errors
- [ ] Test your specific changes manually
- [ ] Check for console errors/warnings
- [ ] Verify responsiveness (mobile/desktop) if UI changes

### Future Testing Goals
- Add unit tests for utility functions
- Add integration tests for complex features
- Set up automated testing in CI/CD pipeline

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

Screenshots (if UI changes)
[Before/after images]

### Commit Message Standards
- Use **Conventional Commits Specification**:
  - `feat: add new feature`
  - `fix: resolve bug`
  - `docs: update documentation`
  - `test: add unit tests`
  - `refactor: improve code structure`
  - `style: formatting changes`
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

## Issue Tracking

### When to Create Issues
- Bug reports with steps to reproduce
- Feature requests with clear requirements
- Technical debt that needs addressing
- Questions or discussions about implementation

### Issue Labels
- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to docs
- `good first issue` - Good for newcomers
- `priority: high/medium/low`

## Definition of Done

A feature is considered complete when:
- [ ] Code is written and tested locally
- [ ] Pull request created with proper description
- [ ] Code review completed and approved
- [ ] All conversations resolved
- [ ] No console errors or warnings
- [ ] Documentation updated (if needed)
- [ ] Branch merged and deleted

## Emergency Procedures
If urgent fixes are needed:
- Still create a pull request
- Review can be expedited but NOT skipped
- Document the urgency in PR description
- Use `hotfix/` branch naming convention

## Conflict Resolution

### Git Merge Conflicts
1. **Communicate immediately** with your partner
2. **Don't resolve conflicts alone** - discuss the changes
3. **Test thoroughly** after resolving conflicts
4. **Document resolution** in commit message

### Workflow Disagreements
1. **Discuss openly** in team meetings
2. **Document decisions** and reasoning
3. **Update guidelines** based on lessons learned
4. **Escalate to mentor** if needed for Orbital project
