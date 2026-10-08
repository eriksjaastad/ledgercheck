# Outputs. WRITE-ONLY reference stack: nothing in this repo applies it.

output "job_name" {
  description = "Start a run with: az containerapp job start -n <job_name> -g <resource_group>."
  value       = azurerm_container_app_job.judge.name
}

output "resource_group" {
  value = azurerm_resource_group.this.name
}

output "job_identity_client_id" {
  value = azurerm_user_assigned_identity.job.client_id
}
