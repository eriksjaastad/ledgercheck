# Resources. WRITE-ONLY reference stack: nothing in this repo applies it.
#
# The workload is a manually triggered Container Apps *job* that runs the
# image's default `ledgercheck judge` and exits. There is no Container App with
# ingress: `ledgercheck serve` binds loopback only and has no login
# (ledgercheck/web.py), so exposing it needs an auth decision first.
#
# Identity: one user-assigned identity pulls the image (AcrPull on
# var.registry_id). No registry passwords.

locals {
  # Secret name -> env var. Keys come from the bool toggles, not from the
  # sensitive values, so for_each never depends on a secret.
  secret_env = merge(
    var.enable_langfuse ? {
      "langfuse-public-key" = "LANGFUSE_PUBLIC_KEY"
      "langfuse-secret-key" = "LANGFUSE_SECRET_KEY"
    } : {},
    var.live_llm ? { "openrouter-api-key" = "OPENROUTER_API_KEY" } : {},
  )
  secret_values = {
    "langfuse-public-key" = var.langfuse_public_key
    "langfuse-secret-key" = var.langfuse_secret_key
    "openrouter-api-key"  = var.openrouter_api_key
  }
  plain_env = merge(
    { LEDGERCHECK_LLM = var.live_llm ? "1" : "0" },
    var.enable_langfuse && var.langfuse_host != null ? { LANGFUSE_HOST = var.langfuse_host } : {},
  )
}

resource "azurerm_resource_group" "this" {
  name     = "${var.name}-rg"
  location = var.location
}

resource "azurerm_log_analytics_workspace" "this" {
  name                = "${var.name}-logs"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_container_app_environment" "this" {
  name                       = "${var.name}-env"
  location                   = azurerm_resource_group.this.location
  resource_group_name        = azurerm_resource_group.this.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.this.id
}

resource "azurerm_user_assigned_identity" "job" {
  name                = "${var.name}-job-id"
  location            = azurerm_resource_group.this.location
  resource_group_name = azurerm_resource_group.this.name
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = var.registry_id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.job.principal_id
}

resource "azurerm_container_app_job" "judge" {
  name                         = "${var.name}-judge"
  location                     = azurerm_resource_group.this.location
  resource_group_name          = azurerm_resource_group.this.name
  container_app_environment_id = azurerm_container_app_environment.this.id
  replica_timeout_in_seconds   = 600
  replica_retry_limit          = 0

  manual_trigger_config {
    parallelism              = 1
    replica_completion_count = 1
  }

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.job.id]
  }

  registry {
    server   = var.registry_server
    identity = azurerm_user_assigned_identity.job.id
  }

  dynamic "secret" {
    for_each = local.secret_env
    content {
      name  = secret.key
      value = local.secret_values[secret.key]
    }
  }

  template {
    container {
      name   = "ledgercheck"
      image  = var.image
      args   = var.args
      cpu    = 0.25
      memory = "0.5Gi"

      dynamic "env" {
        for_each = local.plain_env
        content {
          name  = env.key
          value = env.value
        }
      }

      dynamic "env" {
        for_each = local.secret_env
        content {
          name        = env.value
          secret_name = env.key
        }
      }
    }
  }

  lifecycle {
    precondition {
      condition     = !var.enable_langfuse || (var.langfuse_public_key != null && var.langfuse_secret_key != null)
      error_message = "enable_langfuse needs langfuse_public_key and langfuse_secret_key."
    }
    precondition {
      condition     = !var.live_llm || var.openrouter_api_key != null
      error_message = "live_llm needs openrouter_api_key."
    }
  }

  depends_on = [azurerm_role_assignment.acr_pull]
}
