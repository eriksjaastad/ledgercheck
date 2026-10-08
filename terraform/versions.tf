# Ledger Check on Azure Container Apps. WRITE-ONLY reference stack: nothing in this repo applies it.
#
# This stack is code only and has never been applied. Applying it is your
# decision and creates billable Azure resources. `python3 scripts/deploy.py
# validate` runs `terraform init -backend=false` + `terraform validate` (no
# Azure login, no state, no resources); `scripts/deploy.py apply` refuses.
#
# State is local (no backend block). Choose a remote backend before you apply.

terraform {
  required_version = ">= 1.5"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}
