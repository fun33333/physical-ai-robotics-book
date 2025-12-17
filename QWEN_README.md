# Qwen Code Integration with Speckit Plus

This project is set up with Speckit Plus for specification-driven development and includes integration for Qwen Code alongside Claude Code.

## Setup

The project already includes:

- Speckit Plus configuration in `.specify/`
- Claude Code integration in `.claude/`
- Qwen Code integration in `.qwen/`

## Using Qwen Code

Qwen Code can be used alongside Claude Code with the following Speckit Plus commands:

### Feature Specification Commands
- `/sp.specify [feature description]` - Create/update feature specifications
- `/sp.clarify` - Clarify specification details
- `/sp.plan` - Generate implementation plan
- `/sp.tasks` - Create implementation tasks
- `/sp.checklist` - Generate quality checklists

### Implementation Commands
- `/sp.implement` - Implement features according to tasks
- `/sp.analyze` - Analyze code for improvements
- `/sp.adr` - Create architectural decision records
- `/sp.phr` - Create prompt history records

### Git Integration
- `/sp.git.commit_pr` - Create commits and PRs based on implementation

## How It Works

1. Speckit Plus generates and manages feature specifications in the `specs/` directory
2. Qwen Code uses context from `QWEN.md` for project-specific information
3. Commands are defined in `.qwen/commands/` and mirror those in `.claude/commands/`
4. The system maintains consistency across different AI agents

## Project Context

This is a Docusaurus-based book website for "Physical AI & Humanoid Robotics Book" with:
- Documentation in Markdown/MDX format
- Python code examples (3.10+)
- ROS 2 Humble integration
- Prism for syntax highlighting
- Mermaid for diagrams