# MUDRA UAT Test Plan

## Test Matrix
1. Authentication
- Valid login
- Invalid login
- Role-based admin restrictions

2. Learning & Practice
- Select static gesture, check prediction and feedback
- Select dynamic gesture, verify model_used=dynamic/fusion path
- Record attempt and verify analytics update

3. Quiz
- Start 10-question quiz
- Submit predictions
- Validate final score and attempt entries

4. Analytics
- Refresh attempts list
- Render confusion matrix heatmap
- Validate class counts in matrix cells

5. Admin Model Management
- Refresh registry
- Activate selected version
- Register new version via UI
- Rollback selected family
- Reload predictor and validate active path changed

## Acceptance Criteria
- No crashes during 20-minute continuous usage
- API `/health` and `/ready` return 200 consistently
- Confusion matrix and progress values update after attempts
